import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "2019_ABBY_3_551000_5069000_image_tiles_10x10"
OUTPUT_CSV = DATA_DIR / "baseline_boxes.csv"
DEBUG_DIR = DATA_DIR / "baseline_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Detection parameters
A_MIN, A_MAX = 40, 7000
VAR_THRESHOLD = 90
SAVE_DEBUG = True

# Compute Excess Green index (ExG = 2G - R - B)
def compute_exg(img):
    B, G, R = cv2.split(img.astype(np.int16))
    exg = 2 * G - R - B
    exg = np.clip(exg, 0, 255).astype(np.uint8)
    return exg

# Compute local variance (texture measure)
def compute_texture(gray, ksize=7):
    mean = cv2.boxFilter(gray, ddepth=cv2.CV_32F, ksize=(ksize, ksize))
    mean2 = cv2.boxFilter(gray * gray, ddepth=cv2.CV_32F, ksize=(ksize, ksize))
    var = mean2 - mean * mean
    var = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return var

def detect_boxes(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"WARNING: Could not read {image_path.name}")
        return []

    # Get image height to invert axis later
    height, _ = img.shape[:2]

    # Keep only RGB if image has extra channels (alpha/NIR)
    if len(img.shape) == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    blur = cv2.GaussianBlur(img, (3, 3), 0)
    exg = compute_exg(blur)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    var = compute_texture(gray)

    # Adaptive ExG threshold per image
    t_exg = int(np.mean(exg) + 0.7 * np.std(exg))
    mask_green = cv2.threshold(exg, t_exg, 255, cv2.THRESH_BINARY)[1]
    mask_text = cv2.threshold(var, VAR_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.bitwise_and(mask_green, mask_text)

    # Morphological cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)   #  Area computed here is very wrong
    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if A_MIN <= area <= A_MAX and 0.5 <= w / max(h, 1) <= 2.5:
            boxes.append((x, y, x + w, y + h, area))

    # Save debug visualizations
    if SAVE_DEBUG:
        vis = img.copy()
        for (x1, y1, x2, y2, _) in boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        out_path = DEBUG_DIR / f"{image_path.stem}_boxes.jpg"
        cv2.imwrite(str(out_path), vis)

    return boxes, height

# Main Program
records = []
tif_files = sorted(DATA_DIR.glob("*.tif"))
if not tif_files:
    print(f"ERROR: No tif files found in {DATA_DIR}")
else:
    print(f"Processing {len(tif_files)} tif files")
    for img_path in tif_files:
        boxes, height = detect_boxes(img_path)
        for (x1, y1, x2, y2, area) in boxes:
            records.append({
                "left": x1,
                "bottom": height - y2,
                "right": x2,
                "top": height - y1,
                "score": 1.0,
                "label": "Tree",
                "height": np.nan,
                "area": area,
                "Site": "ABBY",
                "partition_id": img_path.stem
            })
        print(f"{img_path.stem}: {len(boxes)} boxes")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} total boxes to: {OUTPUT_CSV}")
    print(f"Debug images saved in: {DEBUG_DIR}")
