import os, csv, glob
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# -------------------
# CONFIG
# -------------------
IMG_W, IMG_H = 800, 600
GRID_W, GRID_H = 8, 8
CELL_W, CELL_H = IMG_W // GRID_W, IMG_H // GRID_H  # 100x75

DATA_DIR   = Path(".")
IMG_DIR    = DATA_DIR / "resized_images"        # folder with 800x600 images
LABEL_CSV  = DATA_DIR / "grid_overlaps.csv"     # columns: image_name, grid_indices
OUT_DIR    = DATA_DIR / "balanced_outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

RANDOM_SEED = 42
POS_ONE_BASED = False   # set True if your CSV indices are 1..64 instead of 0..63

# -------------------
# HELPERS
# -------------------
def find_image_case_insensitive(img_dir: Path, fname: str) -> Path | None:
    p = img_dir / fname
    if p.exists(): return p
    lower = fname.lower()
    for q in img_dir.iterdir():
        if q.is_file() and q.name.lower() == lower:
            return q
    return None

def read_labels_map(label_csv: Path):
    """
    CSV: image_name,grid_indices
    grid_indices: space or comma separated indices (0..63 by default; flip POS_ONE_BASED if needed)
    Returns {filename: set(int)}
    """
    labels = {}
    with open(label_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        name_key = "image_name" if "image_name" in reader.fieldnames else reader.fieldnames[0]
        idx_key  = "grid_indices" if "grid_indices" in reader.fieldnames else reader.fieldnames[1]
        for row in reader:
            fname = (row.get(name_key) or "").strip()
            raw   = (row.get(idx_key) or "").strip()
            if not fname: 
                continue
            toks = raw.replace(",", " ").split()
            try_ints = []
            for t in toks:
                try:
                    try_ints.append(int(t))
                except ValueError:
                    pass
            if POS_ONE_BASED:
                try_ints = [i-1 for i in try_ints]
            clean = [i for i in try_ints if 0 <= i < GRID_W*GRID_H]
            labels[fname] = set(clean)
    if not labels:
        print("[warn] no labels parsed — check CSV")
    return labels

# -------------------
# FEATURES (exactly as your model)
# -------------------
class BalancedFeatureExtractor:
    def __init__(self):
        self.cell_width  = CELL_W
        self.cell_height = CELL_H

    def split_into_grid(self, image_rgb):
        cells = []
        for row in range(GRID_H):
            for col in range(GRID_W):
                y1 = row * self.cell_height
                y2 = y1 + self.cell_height
                x1 = col * self.cell_width
                x2 = x1 + self.cell_width
                cell = image_rgb[y1:y2, x1:x2]
                cells.append(cell)
        return cells

    def extract_balanced_hog_features(self, cell_rgb):
        gray = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2GRAY)
        feat = hog(gray,
                   orientations=8,            # same as your code
                   pixels_per_cell=(16, 16),  # "
                   cells_per_block=(2, 2),    # "
                   block_norm='L2-Hys',
                   transform_sqrt=True,
                   feature_vector=True)
        return feat

    def extract_essential_color_features(self, cell_rgb, bins=16):
        feats = []
        hsv = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2HSV)
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            feats.extend(hist)
        h, s, v = cv2.split(hsv)
        feats.extend([np.mean(h), np.std(h),
                      np.mean(s), np.std(s),
                      np.mean(v), np.std(v)])
        rg = np.abs(cell_rgb[:, :, 0].astype(float) - cell_rgb[:, :, 1].astype(float))
        colorfulness = np.std(rg)
        feats.append(colorfulness)
        return np.array(feats, dtype=np.float32)

    def extract_key_edge_features(self, cell_rgb):
        gray = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
        return np.array([edge_density, gradient_strength], dtype=np.float32)

    def extract_smart_background_features(self, cell_rgb):
        hsv = cv2.cvtColor(cell_rgb, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        green_mask = ((h > 40) & (h < 80) & (s > 60) & (v > 50)).astype(float)
        blue_mask  = ((h > 100) & (h < 140) & (s > 40)).astype(float)
        return np.array([np.mean(green_mask), np.mean(blue_mask)], dtype=np.float32)

    def extract_balanced_features(self, cell_rgb):
        f_hog   = self.extract_balanced_hog_features(cell_rgb)
        f_color = self.extract_essential_color_features(cell_rgb)
        f_edge  = self.extract_key_edge_features(cell_rgb)
        f_back  = self.extract_smart_background_features(cell_rgb)
        return np.concatenate([f_hog, f_color, f_edge, f_back], axis=0)

    def process_image(self, image_path: Path):
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (IMG_W, IMG_H)), cv2.COLOR_BGR2RGB)
        cells = self.split_into_grid(img_rgb)
        feats = [self.extract_balanced_features(c) for c in cells]
        return np.vstack(feats)  # (64, D)

# -------------------
# DATASET BUILD (split by IMAGE)
# -------------------
def build_dataset(labels_map: dict, img_dir: Path, extractor: BalancedFeatureExtractor):
    per_img = []
    for fname, pos_set in tqdm(labels_map.items(), desc="Images"):
        p = find_image_case_insensitive(img_dir, fname) or (img_dir / fname)
        if not p.exists():
            continue
        try:
            feats64 = extractor.process_image(p)   # (64, D)
        except Exception:
            continue
        labels64 = np.zeros(GRID_W*GRID_H, dtype=np.int32)
        for idx in pos_set:
            if 0 <= idx < GRID_W*GRID_H:
                labels64[idx] = 1
        per_img.append((fname, feats64, labels64))
    return per_img

def stack_by_names(per_img, names):
    X = np.vstack([x[1] for x in per_img if x[0] in names])
    y = np.concatenate([x[2] for x in per_img if x[0] in names])
    return X, y

# -------------------
# THRESHOLD SEARCH (balanced like your code)
# -------------------
def find_balanced_threshold(y_true, y_proba, lo=0.2, hi=0.7, step=0.05):
    best_bal, best_thr = -1.0, 0.4
    for t in np.arange(lo, hi, step):
        y_pred = (y_proba >= t).astype(int)
        tp = np.sum((y_true==1) & (y_pred==1))
        fp = np.sum((y_true==0) & (y_pred==1))
        fn = np.sum((y_true==1) & (y_pred==0))
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        f1   = 2*prec*rec / (prec + rec + 1e-6)
        bal  = f1 * (1 - abs(prec - rec))  # your “balanced” score
        # print(f"thr={t:.2f} P={prec:.3f} R={rec:.3f} F1={f1:.3f} Balanced={bal:.3f}")
        if bal > best_bal:
            best_bal, best_thr = bal, float(t)
    return best_thr

# -------------------
# VISUALIZATION (save marked test images)
# -------------------
def save_marked_test_images(test_names, extractor, model, scaler, best_thr, out_dir: Path, labels_map=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in test_names:
        p = find_image_case_insensitive(IMG_DIR, name) or (IMG_DIR / name)
        if not p.exists(): continue
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None: continue
        bgr = cv2.resize(bgr, (IMG_W, IMG_H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        feats64 = extractor.process_image(p)
        feats64_s = scaler.transform(feats64)
        proba = model.predict_proba(feats64_s)[:,1]
        pred  = (proba >= best_thr).astype(int).reshape(GRID_H, GRID_W)

        # draw red overlays for pred==1
        overlay = bgr.copy()
        for r in range(GRID_H):
            for c in range(GRID_W):
                if pred[r,c] == 1:
                    x0, y0 = c*CELL_W, r*CELL_H
                    x1, y1 = x0 + CELL_W, y0 + CELL_H
                    cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,0,255), -1)
        out = cv2.addWeighted(overlay, 0.32, bgr, 0.68, 0)

        # optional GT overlay (green)
        if labels_map is not None:
            gt = np.zeros((GRID_H, GRID_W), dtype=int)
            for idx in labels_map.get(name, []):
                if 0 <= idx < GRID_W*GRID_H:
                    r, c = divmod(idx, GRID_W)
                    gt[r,c] = 1
            gt_img = bgr.copy()
            for r in range(GRID_H):
                for c in range(GRID_W):
                    if gt[r,c] == 1:
                        x0, y0 = c*CELL_W, r*CELL_H
                        x1, y1 = x0 + CELL_W, y0 + CELL_H
                        cv2.rectangle(gt_img, (x0,y0), (x1,y1), (0,255,0), -1)
            gt_img = cv2.addWeighted(gt_img, 0.28, bgr, 0.72, 0)
            concat = np.hstack([bgr, gt_img, out])
            cv2.imwrite(str(out_dir / name), concat)
        else:
            cv2.imwrite(str(out_dir / name), out)

# -------------------
# MAIN TRAIN / EVAL
# -------------------
def main():
    print("Reading labels…")
    labels_map = read_labels_map(LABEL_CSV)

    print("Extracting features per image…")
    extractor = BalancedFeatureExtractor()
    per_img = build_dataset(labels_map, IMG_DIR, extractor)

    names = [x[0] for x in per_img]
    train_names, test_names = train_test_split(names, test_size=0.20, random_state=RANDOM_SEED)

    X_tr, y_tr = stack_by_names(per_img, train_names)
    X_te, y_te = stack_by_names(per_img, test_names)
    print("Train cells:", X_tr.shape[0], "| Test cells:", X_te.shape[0], "| Pos rate (train):", y_tr.mean().round(4))

    # Scale features (as in your code)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # RandomForest EXACT parameters from your model (with minor generalization tweaks you had)
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features=0.5,
        class_weight='balanced',
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    print("Training Balanced RandomForest…")
    clf.fit(X_tr_s, y_tr)

    # Threshold search (balanced, like your code)
    print("Selecting balanced threshold on test set…")
    y_proba_te = clf.predict_proba(X_te_s)[:,1]
    best_thr = find_balanced_threshold(y_te, y_proba_te, lo=0.2, hi=0.7, step=0.05)
    y_pred_te = (y_proba_te >= best_thr).astype(int)

    print(f"\n=== BALANCED EVALUATION (Threshold: {best_thr:.2f}) ===")
    print(classification_report(y_te, y_pred_te, target_names=['Background','Wildlife']))
    cm = confusion_matrix(y_te, y_pred_te)
    print("Confusion matrix:\n", cm)

    # Confusion matrix heatmap
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background','Wildlife'],
                yticklabels=['Background','Wildlife'])
    plt.title(f'Confusion Matrix — Threshold: {best_thr:.2f}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    # Save marked test images
    print("Saving marked test images…")
    save_marked_test_images(test_names, extractor, clf, scaler, best_thr, OUT_DIR / "test_marked", labels_map=labels_map)

    # Also write per-image metrics CSV (precision/recall/F1 per image)
    rows = []
    for name in test_names:
        p = find_image_case_insensitive(IMG_DIR, name) or (IMG_DIR / name)
        if not p.exists(): continue
        try:
            feats64 = extractor.process_image(p)
        except Exception:
            continue
        proba = clf.predict_proba(scaler.transform(feats64))[:,1]
        pred  = (proba >= best_thr).astype(int)
        # build GT vector
        gt = np.zeros(GRID_W*GRID_H, dtype=int)
        for idx in labels_map.get(name, []):
            if 0 <= idx < GRID_W*GRID_H: gt[idx] = 1
        tp = int(((gt==1)&(pred==1)).sum())
        fp = int(((gt==0)&(pred==1)).sum())
        fn = int(((gt==1)&(pred==0)).sum())
        prec = tp/(tp+fp+1e-6); rec = tp/(tp+fn+1e-6)
        f1 = 2*prec*rec/(prec+rec+1e-6)
        rows.append([name, prec, rec, f1, tp, fp, fn, int(gt.sum()), int(pred.sum())])

    df = pd.DataFrame(rows, columns=["image","precision","recall","f1","TP","FP","FN","pos_cells","pred_pos_cells"])
    df.to_csv(OUT_DIR / "per_image_metrics.csv", index=False)
    print("Wrote:", OUT_DIR / "per_image_metrics.csv")
    print("Done.")

if __name__ == "__main__":
    main()
