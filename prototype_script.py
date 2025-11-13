import os
import cv2
import numpy as np
import rasterio
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# DDirectory Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "2019_ABBY_3_551000_5069000_image_tiles_10x10"
CSV_PATH = DATA_DIR / "ABBY_2019_with_partitions.csv"
OUTPUT_CSV = DATA_DIR / "cnn_boxes.csv"
MODEL_PATH = DATA_DIR / "texture_cnn_model.pth"
DEBUG_DIR = DATA_DIR / "cnn_debug"
CROP_DEBUG = DATA_DIR / "crop_debug"
DEBUG_DIR.mkdir(exist_ok=True)
CROP_DEBUG.mkdir(exist_ok=True)

# Hyperparameter Setup and Settings
PATCH_SIZE = 48
STRIDE = 8
THRESHOLD_PERCENTILE = 40
NEG_RATIO = 3.0
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = [0.485, 0.456, 0.406, 0.5]
STD = [0.229, 0.224, 0.225, 0.25]

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Parse CSV and filtering data
df = pd.read_csv(CSV_PATH, dtype={'partition_id': str})
# Technically partitioning sets all non plot processed partition id to -1 already but just in case 
df.loc[~(df["geo_index"] == "551000_5069000"), "partition_id"] = "-1" 
df = df[df["partition_id"] != "-1"]
tile_ids = df["partition_id"].unique()
print(f"Loaded {len(df)} crowns across {len(tile_ids)} tiles")

#  Georeferencing the tiles
def get_tile_bounds(tile_id: str):
    tile_num = int(tile_id[1:]) # offset fix, pretty sure I fixed the offset when the partitions became 0-99 but we'll see
    img_path = DATA_DIR / f"P{tile_num:02d}.tif"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing image for {tile_id}")
    with rasterio.open(img_path) as src:
        bounds = src.bounds
    return bounds.left, bounds.right, bounds.bottom, bounds.top

# Splitting the training and validation sets
np.random.seed(42)
np.random.shuffle(tile_ids)
split_idx = int(len(tile_ids) * 0.8)
train_tiles = tile_ids[:split_idx]
val_tiles = tile_ids[split_idx:]
print(f"Training tiles: {len(train_tiles)} | Validation tiles: {len(val_tiles)}")

# Add texture to the image
def add_texture_channel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex = np.abs(lap)
    tex = tex / (tex.max() + 1e-5)
    tex = np.expand_dims(tex, 2)
    img = img.astype(np.float32) / 255.0
    return np.concatenate([img, tex], axis=2)

# Dataset class for going through the patches
class TreePatchDataset(Dataset):
    # TODO: add type annotation, cba going back every time
    def __init__(self, tiles, df, patch_size=64, neg_ratio=1.0, debug_save=False):
        self.samples = []
        total_pos, total_neg = 0, 0
        dbg = 0

        for i, tile_id in enumerate(tiles):
            print(f"[{i+1}/{len(tiles)}] Processing {tile_id}...")
            try:
                left, right, bottom, top = get_tile_bounds(tile_id)
                tile_num = int(tile_id[1:]) + 1             # Have to slice after 1 because it's P## btw
                img_path = DATA_DIR / f"P{tile_num}.tif"
                if not img_path.exists():
                    print(f"Missing {img_path.name}")
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Could not read {img_path.name}")
                    continue

                h, w = img.shape[:2]
                boxes = df[df["partition_id"] == tile_id][["left", "top", "right", "bottom"]].values

                # If there are no boxes, make only negative samples
                if len(boxes) == 0:
                    for _ in range(int(neg_ratio * 10)):
                        rx = np.random.randint(0, w - patch_size)
                        ry = np.random.randint(0, h - patch_size)
                        crop = img[ry:ry + patch_size, rx:rx + patch_size]
                        crop = add_texture_channel(crop)
                        crop = (crop - np.array(MEAN)) / np.array(STD)
                        self.samples.append((torch.tensor(crop.transpose(2, 0, 1), dtype=torch.float32),
                                             torch.tensor(0.0)))
                    continue

                pos_total, neg_total = 0, 0

                # Making positive samples
                for (xw1, yw_top, xw2, yw_bot) in boxes:
                    if (xw2 < left or xw1 > right or yw_bot < bottom or yw_top > top):
                        continue

                    x1 = int(((xw1 - left) / (right - left)) * w)
                    x2 = int(((xw2 - left) / (right - left)) * w)
                    y1 = int(((top - yw_top) / (top - bottom)) * h)
                    y2 = int(((top - yw_bot) / (top - bottom)) * h)
                    x1, x2 = sorted([max(0, x1), min(w - 1, x2)])
                    y1, y2 = sorted([max(0, y1), min(h - 1, y2)])

                    if (x2 - x1) < 4 or (y2 - y1) < 4:
                        continue

                    crop = img[y1:y2, x1:x2]
                    crop = cv2.resize(crop, (patch_size, patch_size))
                    crop = add_texture_channel(crop)
                    crop = (crop - np.array(MEAN)) / np.array(STD)
                    self.samples.append((torch.tensor(crop.transpose(2, 0, 1), dtype=torch.float32),
                                         torch.tensor(1.0)))
                    pos_total += 1
                    total_pos += 1

                    if debug_save and dbg < 10:
                        cv2.imwrite(str(CROP_DEBUG / f"{tile_id}_pos_{dbg}.jpg"), (crop[:, :, :3] * 255).astype(np.uint8))
                        dbg += 1

                # Making negative samples
                num_neg = max(1, int(len(boxes) * neg_ratio))
                max_attempts = num_neg * 10
                attempts = 0
                while neg_total < num_neg and attempts < max_attempts:
                    rx = np.random.randint(0, w - patch_size)
                    ry = np.random.randint(0, h - patch_size)
                    crop_box = (rx, ry, rx + patch_size, ry + patch_size)

                    overlaps = any(
                        (x1 < crop_box[2] and x2 > crop_box[0] and y1 < crop_box[3] and y2 > crop_box[1])
                        for (x1, y1, x2, y2) in [
                            (
                                int(((xw1 - left) / (right - left)) * w),
                                int(((top - yw_top) / (top - bottom)) * h),
                                int(((xw2 - left) / (right - left)) * w),
                                int(((top - yw_bot) / (top - bottom)) * h),
                            )
                            for (xw1, yw_top, xw2, yw_bot) in boxes
                        ]
                    )

                    if overlaps:
                        attempts += 1
                        continue

                    crop = img[ry:ry + patch_size, rx:rx + patch_size]
                    if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
                        attempts += 1
                        continue

                    crop = add_texture_channel(crop)
                    crop = (crop - np.array(MEAN)) / np.array(STD)
                    self.samples.append((torch.tensor(crop.transpose(2, 0, 1), dtype=torch.float32),
                                         torch.tensor(0.0)))
                    neg_total += 1
                    total_neg += 1
                    attempts += 1

                if attempts >= max_attempts:
                    print(f"Gave up generating negatives for {tile_id} after {attempts} attempts, only ({neg_total} generated)")

                print(f"[{tile_id}] Pos: {pos_total}, Neg: {neg_total}")

            except Exception as e:
                print(f"Error on {tile_id}: {e}")
                continue

        np.random.shuffle(self.samples)
        print(f"Total Pos: {total_pos}, Total Neg: {total_neg}, Total Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

# Making the CNN model
class TextureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = TextureCNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training the model
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Loaded saved model.")
else:
    train_ds = TreePatchDataset(train_tiles, df, PATCH_SIZE, NEG_RATIO, debug_save=True)
    val_ds = TreePatchDataset(val_tiles, df, PATCH_SIZE, NEG_RATIO)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                preds = model(imgs)
                correct += ((preds > 0.5) == (labels > 0.5)).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss={total_loss/len(train_loader):.4f} | Val Acc={acc:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# Detecting bounding boxes using the trained model
def detect_boxes(img, model, img_path=None):
    h, w = img.shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    model.eval()

    # Sliding window to make probability map
    with torch.no_grad():
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch4 = add_texture_channel(patch)
                patch4 = (patch4 - np.array(MEAN)) / np.array(STD)
                patch4 = torch.tensor(patch4.transpose(2, 0, 1),
                                      dtype=torch.float32, device=DEVICE).unsqueeze(0)
                p = model(patch4).item()
                prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += p
                counts[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    prob_map /= np.maximum(counts, 1e-5)

    # Making the overlay map
    overlay_color = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_blend = cv2.addWeighted(img, 0.6, overlay_color, 0.4, 0)
    overlay_path = DEBUG_DIR / f"{img_path.stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay_blend)

    # Making the mask map
    blurred = cv2.GaussianBlur(prob_map, (3, 3), 0)
    cutoff = np.percentile(blurred, THRESHOLD_PERCENTILE)
    mask = (blurred <= cutoff).astype(np.uint8) * 255
    mask_path = DEBUG_DIR / f"{img_path.stem}_mask.jpg"
    cv2.imwrite(str(mask_path), mask)

    # Finding contours and bounding boxes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    vis = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 5000:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        radius = int(np.sqrt(area / np.pi))
        radius = np.clip(radius, 5, 50)

        if area > 1500:
            submask = np.zeros_like(mask)
            cv2.drawContours(submask, [cnt], -1, 255, -1)
            dist = cv2.distanceTransform(submask, cv2.DIST_L2, 5)
            local_max = cv2.dilate(dist, np.ones((7, 7), np.uint8))
            peaks = (dist >= (local_max - 1e-5)) & (dist > 0.3 * dist.max())
            ys, xs = np.where(peaks > 0)
            for (px, py) in zip(xs, ys):
                x1, y1 = max(0, px - radius), max(0, py - radius)
                x2, y2 = min(w - 1, px + radius), min(h - 1, py + radius)
                boxes.append((x1, y1, x2, y2, area))
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
            boxes.append((x1, y1, x2, y2, area))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Save bounding box visualization
    box_vis_path = DEBUG_DIR / f"{img_path.stem}_cnn_boxes.jpg"
    cv2.imwrite(str(box_vis_path), vis)

    print(f"{img_path.name}: {len(boxes)} variable-size crowns")
    print(f"  ├─ Overlay saved: {overlay_path.name}")
    print(f"  ├─ Mask saved:    {mask_path.name}")
    print(f"  └─ Boxes saved:   {box_vis_path.name}")
    return boxes

# Running detection on all images and saving results
records = []
tifs = sorted(list(DATA_DIR.glob("*.tif")))
print(f"{len(tifs)} images were found for detection.")
for img_path in tifs:
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    boxes = detect_boxes(img, model, img_path)
    print(f"Detected {len(boxes)} crowns in {img_path.name}")
    vis = img.copy()
    for (x1, y1, x2, y2, _) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(DEBUG_DIR / f"{img_path.stem}_cnn_boxes.jpg"), vis)
    for (x1, y1, x2, y2, area) in boxes:
        records.append({
            "left": x1, "bottom": y2, "right": x2, "top": y1,
            "score": 1.0, "label": "Tree", "height": np.nan, "area": area,
            "Site": "ABBY", "partition_id": img_path.stem
        })

pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(records)} boxes to {OUTPUT_CSV}")