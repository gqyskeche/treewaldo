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
import timeit
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "partition_output"
CSV_PATH = DATA_DIR / "a_trees_with_partitions.csv"
OUTPUT_CSV = DATA_DIR / "cnn_boxes.csv"
MODEL_PATH = DATA_DIR / "texture_cnn_model.pth"
DEBUG_DIR = DATA_DIR / "cnn_debug"
CROP_DEBUG = DATA_DIR / "crop_debug"
DEBUG_DIR.mkdir(exist_ok=True)
CROP_DEBUG.mkdir(exist_ok=True)

# Hyperparameter Setup and Settings
PATCH_SIZE = 48         # This is the size of the bounding box btw 
STRIDE = 10
THRESHOLD_PERCENTILE = 40
NEG_RATIO = 3.0
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
force_rerun = True

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Add texture to the image to differententiate between grass and tree
# I don't know how much this helps tbh but the entire picture can be green such as in our dataset
# So this is (probably) helpful
def add_texture_channel(img):
    img = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex = np.abs(lap)

    # Scale texture to 0–1 using a stable constant, not local max
    tex = tex / 255.0

    tex = np.expand_dims(tex, 2)
    return np.concatenate([img, tex], axis=2)


# Time Start
start = timeit.default_timer()

# Compute RGB mean and std across entire dataset
image_list = [DATA_DIR / f"P{i}.tif" for i in range(100)]
pixel_sum = np.zeros(4, dtype=np.float64)
pixel_sq_sum = np.zeros(4, dtype=np.float64)
pixel_count = 0
samples_per_image=500

for p in image_list:
    img = cv2.imread(str(p)).astype(np.float32)
    h, w = img.shape[:2]
    # Texture is patch based so we take a large sample to compute statistics instead
    for _ in range(samples_per_image):
        rx = np.random.randint(0, w - PATCH_SIZE)
        ry = np.random.randint(0, h - PATCH_SIZE)

        crop = img[ry:ry+PATCH_SIZE, rx:rx+PATCH_SIZE]
        crop = add_texture_channel(crop)
        crop = crop.reshape(-1, 4)

        pixel_sum += crop.sum(axis=0)
        pixel_sq_sum += (crop**2).sum(axis=0)
        pixel_count += crop.shape[0]

mean = pixel_sum / pixel_count
var = pixel_sq_sum / pixel_count - mean**2  # E[X^2] - (E[X])^2
std = np.sqrt(np.maximum(var, 1e-8))
MEAN, STD = mean.tolist(), std.tolist()
print("Mean(RGB, Texture):", MEAN)
print("STD(RGB, Texture):", STD)

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
    img_path = DATA_DIR / f"P{tile_num}.tif"
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
                tile_num = int(tile_id[1:])             # Have to slice after 1 because it's P## btw
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
                # This shouldn't be possible given that every partition so far has bbox
                # but here in case we train on different data
                if len(boxes) == 0:
                    for _ in range(int(neg_ratio * 10)):
                        # Generates random patch of patchsize somewhere on the tif
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
                        print("OUT OF BOUNDS, DOUBLE CHECK PARTITIONING")
                        continue

                    # Relative percentage on UTM coords along axis to Pixel coordinates
                    x1 = int(((xw1 - left) / (right - left)) * w)
                    x2 = int(((xw2 - left) / (right - left)) * w)
                    y1 = int(((top - yw_top) / (top - bottom)) * h)
                    y2 = int(((top - yw_bot) / (top - bottom)) * h)
                    x1, x2 = sorted([max(0, x1), min(w - 1, x2)])
                    y1, y2 = sorted([max(0, y1), min(h - 1, y2)])

                    if (x2 - x1) < 4 or (y2 - y1) < 4:
                        continue

                    # Compute overlap between GT bounding box and patch
                    patch_area = patch_size * patch_size

                    gt_x1, gt_y1 = x1, y1
                    gt_x2, gt_y2 = x2, y2

                     # Patch overlaps with reference bbox
                    rx_min = max(0, gt_x2 - patch_size)
                    rx_max = min(gt_x1, w - patch_size)
                    ry_min = max(0, gt_y2 - patch_size)
                    ry_max = min(gt_y1, h - patch_size)

                    if rx_min > rx_max or ry_min > ry_max:
                        # If box smaller than patch, center patch on GT
                        rx = max(0, min(w - patch_size, gt_x1 - patch_size // 2))
                        ry = max(0, min(h - patch_size, gt_y1 - patch_size // 2))
                    else:
                        rx = np.random.randint(rx_min, rx_max + 1)
                        ry = np.random.randint(ry_min, ry_max + 1)

                    patch_global_x1, patch_global_y1 = rx, ry
                    patch_global_x2, patch_global_y2 = rx + patch_size, ry + patch_size

                    # Intersection with GT bbox
                    ix1 = max(patch_global_x1, gt_x1)
                    iy1 = max(patch_global_y1, gt_y1)
                    ix2 = min(patch_global_x2, gt_x2)
                    iy2 = min(patch_global_y2, gt_y2)

                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    inter_area = iw * ih

                    # Soft label (fraction of patch covered by tree)
                    label = min(inter_area / patch_area, 1.0)


                    # After calculating label, build the crop
                    crop = img[patch_global_y1:patch_global_y2, patch_global_x1:patch_global_x2]
                    crop = add_texture_channel(crop)
                    crop = (crop - np.array(MEAN)) / np.array(STD)

                    self.samples.append((
                        torch.tensor(crop.transpose(2, 0, 1), dtype=torch.float32),
                        torch.tensor(label, dtype=torch.float32)
                    ))
                    pos_total += 1
                    total_pos += 1

                    if debug_save and dbg < 10:
                        cv2.imwrite(str(CROP_DEBUG / f"{tile_id}_pos_{dbg}.jpg"), (crop[:, :, :3] * 255).astype(np.uint8))
                        dbg += 1

                # Making negative samples
                num_neg = max(1, int(len(boxes) * neg_ratio))   # off the top of my head, this should be max(1, 230?)
                max_attempts = num_neg * 10                     # ~2300
                attempts = 0
                overlap_counter = 0
                bad_crop_counter = 0
                while neg_total < num_neg and attempts < max_attempts:
                    rx = np.random.randint(0, w - patch_size)
                    ry = np.random.randint(0, h - patch_size)
                    crop_box = (rx, ry, rx + patch_size, ry + patch_size)
                    # Bottom left is (0, 0), tested formula against all 6 scenarios, works
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
                        overlap_counter += 1
                        continue

                    crop = img[ry:ry + patch_size, rx:rx + patch_size]
                    if crop.shape[0] != patch_size or crop.shape[1] != patch_size:
                        attempts += 1
                        bad_crop_counter += 1
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
                    print(f"Overlaps Found: {overlap_counter} | Bad Crops Found: {bad_crop_counter}")

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
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)   # changed to raw logits now due to loss f change

model = TextureCNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training the model
if MODEL_PATH.exists() and not force_rerun:
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
            preds_logits = model(imgs)
            loss = criterion(preds_logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                preds = torch.sigmoid(model(imgs))
                correct += ((preds > 0.5) == (labels > 0.5)).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss={total_loss/len(train_loader):.4f} | Val Acc={acc:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

train_time = timeit.default_timer()

# Detecting bounding boxes using the trained model
def detect_boxes(img, model, img_path=None):
    h, w = img.shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    model.eval()

    # Sliding window to make probability map
    pmap_x, pmap_y = 0, 0
    with torch.no_grad():
        while pmap_y <= h - PATCH_SIZE:
            while pmap_x <= w - PATCH_SIZE:
                patch = img[pmap_y:pmap_y+PATCH_SIZE, pmap_x:pmap_x+PATCH_SIZE]
                patch4 = add_texture_channel(patch)
                patch4 = (patch4 - np.array(MEAN)) / np.array(STD)
                patch4 = torch.tensor(patch4.transpose(2, 0, 1),
                                      dtype=torch.float32, device=DEVICE).unsqueeze(0)
                p = torch.sigmoid(model(patch4)).item()
                # All these patches are overlapping and superimposing probabilities I believe
                prob_map[pmap_y:pmap_y+PATCH_SIZE, pmap_x:pmap_x+PATCH_SIZE] += p
                counts[pmap_y:pmap_y+PATCH_SIZE, pmap_x:pmap_x+PATCH_SIZE] += 1
                pmap_x += STRIDE
            pmap_x = 0
            pmap_y += STRIDE


    prob_map /= np.maximum(counts, 1e-5)

    # Making the overlay map
    overlay_color = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay_blend = cv2.addWeighted(img, 0.6, overlay_color, 0.4, 0)
    overlay_path = DEBUG_DIR / f"{img_path.stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay_blend)

    # Making the mask map
    blurred = cv2.GaussianBlur(prob_map, (3, 3), 0)
    cutoff = np.percentile(blurred, THRESHOLD_PERCENTILE)
    mask = (blurred >= cutoff).astype(np.uint8) * 255
    mask_path = DEBUG_DIR / f"{img_path.stem}_mask.jpg"
    cv2.imwrite(str(mask_path), mask)

##############
    # Boxes too large initially
    # Watershed segmentation
    # Binary mask
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    bin_mask = (bin_mask > 0).astype(np.uint8)

    # Distance transform
    dist = distance_transform_edt(bin_mask)

    # Peak detection (new API)
    coords = peak_local_max(
        dist,
        min_distance=8,
        threshold_rel=0.1,
        exclude_border=False
    )

    # Marker image
    markers = np.zeros(dist.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    # Watershed on NEGATIVE distance transform
    labels = watershed(-dist, markers, mask=bin_mask)

    # Extract bounding boxes
    boxes = []
    for region_id in np.unique(labels):
        if region_id == 0:
            continue

        ys, xs = np.where(labels == region_id)
        if len(xs) < 30:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        area = (x2 - x1) * (y2 - y1)

        boxes.append((x1, y1, x2, y2, area))    


###############
    # Boxes too large rn as well, even bigger than watershed
    # Local Maxima for on mask peaks method
    # Ensure binary 0/255
    # _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # # Distance transform (creates crown peaks)
    # dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)

    # # Normalize
    # dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # # Find local maxima
    # peaks = (dist_norm > 0.35).astype(np.uint8)  # tweak threshold
    # peaks = cv2.dilate(peaks, np.ones((3,3), np.uint8), iterations=1)

    # # Label peaks → each label corresponds to one crown
    # num_labels, labels = cv2.connectedComponents(peaks)

    # boxes = []
    # for i in range(1, num_labels):
    #     ys, xs = np.where(labels == i)
    #     if len(xs) < 10: continue   # skip noise

    #     x1, x2 = xs.min(), xs.max()
    #     y1, y2 = ys.min(), ys.max()
    #     area = (x2 - x1) * (y2 - y1)
    #     boxes.append((x1, y1, x2, y2, area))


###############
    # Contour Method

    # # Finding contours and bounding boxes
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # boxes = []
    # vis = img.copy()

    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area < 20:
    #         continue

    #     x, y, bw, bh = cv2.boundingRect(cnt)
    #     radius = int(np.sqrt(area / np.pi))
    #     radius = np.clip(radius, 5, 50)

    #     if area > 1500:
    #         submask = np.zeros_like(mask)
    #         cv2.drawContours(submask, [cnt], -1, 255, -1)
    #         dist = cv2.distanceTransform(submask, cv2.DIST_L2, 5)
    #         local_max = cv2.dilate(dist, np.ones((7, 7), np.uint8))
    #         thresh = 0.35 * dist.max()
    #         peaks = (dist == local_max) & (dist >= thresh)
    #         ys, xs = np.where(peaks)
    #         for (px, py) in zip(xs, ys):
    #             x1, y1 = max(0, px - radius), max(0, py - radius)
    #             x2, y2 = min(w - 1, px + radius), min(h - 1, py + radius)
    #             boxes.append((x1, y1, x2, y2, area))
    #             cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     else:
    #         x1, y1 = max(0, x), max(0, y)
    #         x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
    #         boxes.append((x1, y1, x2, y2, area))
    #         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Moved this logic to the very end after NMS for now
    # Save bounding box visualization
    box_vis_path = DEBUG_DIR / f"{img_path.stem}_cnn_boxes.jpg"
    # cv2.imwrite(str(box_vis_path), vis)

    print(f"{img_path.name}: {len(boxes)} variable-size crowns")
    print(f"  ├─ Overlay saved: {overlay_path.name}")
    print(f"  ├─ Mask saved:    {mask_path.name}")
    print(f"  └─ Boxes saved:   {box_vis_path.name}")
    return boxes


# Too many close overlapping boxes, could increase stride but too arbitrary
# so we apply non maximimum suppression post-processing
# Could do with CV2 NMS or PyTorch NMS look into it
def nms(boxes, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    
    boxes_array = np.array([b[:4] for b in boxes], dtype=np.float32)  # only x1,y1,x2,y2
    areas = (boxes_array[:,2] - boxes_array[:,0]) * (boxes_array[:,3] - boxes_array[:,1])
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes_array[i,0], boxes_array[order[1:],0])
        yy1 = np.maximum(boxes_array[i,1], boxes_array[order[1:],1])
        xx2 = np.minimum(boxes_array[i,2], boxes_array[order[1:],2])
        yy2 = np.minimum(boxes_array[i,3], boxes_array[order[1:],3])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    # Return original box tuples (with area) for the kept indices
    return [boxes[i] for i in keep]

# Running detection on all images and saving results
# records = []
# tifs = sorted(list(DATA_DIR.glob("*.tif")))
# print(f"{len(tifs)} images were found for detection.")
# for img_path in tifs:
#     img = cv2.imread(str(img_path))
#     if img is None:
#         continue
#     boxes = detect_boxes(img, model, img_path)
#     boxes = nms(boxes, iou_threshold=0.1)
#     print(f"Detected {len(boxes)} crowns in {img_path.name}")

#     # Drawing rectangles
#     vis = img.copy()
#     for (x1, y1, x2, y2, _) in boxes:
#         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     cv2.imwrite(str(DEBUG_DIR / f"{img_path.stem}_cnn_boxes.jpg"), vis)

#     # Adding to csv list
#     # TODO: implement confidence
#     for (x1, y1, x2, y2, area) in boxes:
#         records.append({
#             "left": x1, "bottom": 1000-y2, "right": x2, "top": 1000-y1,
#             "score": 1.0, "label": "Tree", "height": np.nan, "area": area,
#             "Site": "ABBY", "partition_id": img_path.stem
#         })

# pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
# print(f"\nSaved {len(records)} boxes to {OUTPUT_CSV}")

end = timeit.default_timer()
print(f"Training Time: {train_time - start:.2f}")
print(f"Map Generation and Classification Time: {end - train_time:.2f}")
print(f"Time elapsed: {end - start:.2f}")
