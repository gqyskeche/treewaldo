import numpy as np
import pandas as pd
import cv2
import rasterio
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import timeit
from tqdm import tqdm
import torch.nn.functional as F

# Input and Output Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "partition_output"
INPUT_DIR = BASE_DIR / "input_photos"
DATA_DIR.mkdir(exist_ok=True)
CSV_PATH = BASE_DIR / "trees_with_partitions_multi_image.csv"   # "a_trees_with_partitions.csv"(only for 506900) | "trees_with_partitions_multi_image.csv"(all 10 photos, 1000 partitions)
MODEL_PATH = DATA_DIR / "unet_tree_seg.pth"
SEG_DEBUG_DIR = DATA_DIR / "seg_debug"
SEG_DEBUG_DIR.mkdir(exist_ok=True)

# Constants, HP, variables
EPOCHS = 5
BATCH_SIZE = 4
LR = 1e-3
PATCH_SIZE = 48   

# Let's play.....Do you have a GPU?(in a Barney Stinson voice)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
force_rerun = True

print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# Throw this down there later
start = timeit.default_timer()



# Add texture to the image to differententiate between grass and tree
# I don't know how much this helps tbh but the entire picture can be green such as in our dataset
# So this is (probably) helpful
def add_texture_channel(img):
    img = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    tex = np.abs(lap)
    tex = tex / 255.0       # Scale texture to 0–1 using a stable constant, not local max
    tex = np.expand_dims(tex, 2)
    return np.concatenate([img, tex], axis=2)

# Compute RGB mean and std across entire dataset
image_list = sorted(INPUT_DIR.glob("P*.tif"))
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
# df.loc[~(df["geo_index"] == "551000_5069000"), "partition_id"] = "-1" 
# df = df[df["partition_id"] != "-1"]

# Multi plot execution
df = df[df["partition_id"] != "-1"]

preprocess_time = timeit.default_timer()

tile_ids = df["partition_id"].unique()
print(f"Loaded {len(df)} crowns across {len(tile_ids)} tiles")

# # Get all tiles
# all_tiles = sorted([p.stem for p in INPUT_DIR.glob("P*.tif")])   # e.g. ['P0', 'P1', ..., 'P999']

# print(len(all_tiles), all_tiles[:10])

# # Get all tiles with crowns in t he partition
# csv_tiles = sorted(tile_ids.tolist())
# print(len(csv_tiles), csv_tiles[:10])

# # Take set difference to see actual partition with no crowns
# missing_tiles = sorted(set(all_tiles) - set(csv_tiles))
# print("Tiles with no crowns in CSV:", missing_tiles)
# for t in missing_tiles:
#     print(t, len(df[df["partition_id"] == t]))
# raise KeyError

def boxes_to_mask(tile_id: str, df_tile):
    """
    Returns a binary mask (H, W) with 1 inside tree boxes, 0 elsewhere.
    """

    tile_num = int(tile_id[1:])  # assumes tile_id like 'P0', 'P37'
    img_path = INPUT_DIR / f"P{tile_num}.tif"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing image for {tile_id}")

    with rasterio.open(img_path) as src:
        left, bottom, right, top = src.bounds       # NOTE: rasterio order, don't change ts
        h, w = src.height, src.width
        
        crs = src.crs
        transform = src.transform

    kept = 0
    skipped_small = 0


    mask = np.zeros((h, w), dtype=np.uint8)

    for i, (_, row) in enumerate(df_tile.iterrows()):
        xw1, yw_bottom, xw2, yw_top = row["left"], row["bottom"], row["right"], row["top"]

        # Use rasterio to map world -> pixel (row, col)
        # Note: src.index(x, y) => (row, col)
        row1, col1 = rasterio.transform.rowcol(transform, xw1, yw_top)
        row2, col2 = rasterio.transform.rowcol(transform, xw2, yw_bottom)

        # Convert to y,x and clamp to image bounds
        y1, y2 = sorted([max(0, row1), min(h - 1, row2)])
        x1, x2 = sorted([max(0, col1), min(w - 1, col2)])

        if (x2 - x1) < 1 or (y2 - y1) < 1:
            skipped_small += 1
            continue

        # Shrinkage method to reduce ground detection(doens't work)
        # shrink = 0.0  # 20% shrink on each side, tune 0.15–0.3

        # bw = x2 - x1   # box width
        # bh = y2 - y1   # box height

        # x1s = int(x1 + shrink * bw)
        # x2s = int(x2 - shrink * bw)
        # y1s = int(y1 + shrink * bh)
        # y2s = int(y2 - shrink * bh)

        # if x2s <= x1s or y2s <= y1s:
        #     continue  # skip degenerate

        # mask[y1s:y2s, x1s:x2s] = 1

        mask[y1:y2, x1:x2] = 1
        kept += 1

        # Don't de comment these unless you really need to debug and want upload TQDM bars
        # Print the first few boxes for inspection
        # if i < 3:
        #     print(
        #         f"[{tile_id}] box {i}: "
        #         f"world=({xw1:.1f},{yw_bottom:.1f})–({xw2:.1f},{yw_top:.1f}), "
        #         f"px=({x1},{y1})–({x2},{y2})"
        #     )

    # print(
    #     f"[{tile_id}] boxes kept={kept}, "
    #     f"skipped_small={skipped_small}, "
    #     f"mask_sum={mask.sum()}"
    # )

    return mask



class TreeSegTileDataset(Dataset):
    def __init__(self, tile_ids, df, augment=False):
        self.tile_ids = list(tile_ids)
        self.df = df
        self.augment = augment

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        tile_num = int(tile_id[1:])
        img_path = INPUT_DIR / f"P{tile_num}.tif"

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Could not read {img_path}")

        mask = boxes_to_mask(tile_id, self.df[self.df["partition_id"] == tile_id])

        # optional augmentation
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()

        # add texture channel + normalize with global MEAN/STD
        img4 = add_texture_channel(img)              # (H, W, 4)
        img4 = (img4 - np.array(MEAN)) / np.array(STD)
        img4 = img4.transpose(2, 0, 1)               # (C, H, W)

        img_tensor = torch.tensor(img4, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return img_tensor, mask_tensor, tile_id

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super().__init__()
        self.enc1 = self.block(in_ch, 32)
        self.enc2 = self.block(32, 64)
        self.enc3 = self.block(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.dec2 = self.block(128 + 64, 64)
        self.dec1 = self.block(64 + 32, 32)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)

        self.out = nn.Conv2d(32, out_ch, 1)

    def block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.out(d1)
        return logits

class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # BCE with logits, foreground weighted
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

        # Dice loss
        probs = torch.sigmoid(logits)
        smooth = 1.0

        probs_flat   = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union        = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)

        return bce + dice.mean()


# Splitting the training and validation sets
np.random.seed(42)
np.random.shuffle(tile_ids)
split_idx = int(len(tile_ids) * 0.8)
train_tiles = tile_ids[:split_idx]
val_tiles = tile_ids[split_idx:]
print(f"Training tiles: {len(train_tiles)} | Validation tiles: {len(val_tiles)}")


model = SimpleUNet(in_ch=4, out_ch=1).to(DEVICE)
pos_weight = torch.tensor([3.0], device=DEVICE)
criterion = BCEDiceLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Model Training/Loading
if MODEL_PATH.exists() and not force_rerun:
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print("Loaded existing U-Net model.")
else:
    train_ds = TreeSegTileDataset(train_tiles, df, augment=True)
    val_ds   = TreeSegTileDataset(val_tiles, df, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, masks, _ in tqdm(train_loader,
                                   desc=f"Epoch {epoch+1}/{EPOCHS} [train]",
                                   leave=False):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # simple validation IoU
        model.eval()
        ious = []
        with torch.no_grad():
            tiles_with_union = 0
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(DEVICE)
                masks = masks.to(DEVICE)
                logits = model(imgs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.3).float()
                inter = (preds * masks).sum().item()
                union = ((preds + masks) > 0).float().sum().item()
                if union > 0:
                    ious.append(inter / union)
                    tiles_with_union += 1

        mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss={total_loss/len(train_loader):.4f} | "
              f"Val IoU={np.mean(ious):.3f}")
        print(f"Epoch {epoch+1}: tiles with non-empty union in val = {tiles_with_union}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved U-Net model to {MODEL_PATH}")

train_time = timeit.default_timer()

# the min and max tree regions are hardcoded
def segment_and_boxes_on_tile(model, img_path: Path, tile_id: str,
                              prob_thresh=0.6,
                              min_region_pixels=120,
                              max_region_pixels=10000,
                              min_distance=2,
                              dt_thresh_rel=0.05):

    img = cv2.imread(str(img_path))

    # build input
    img4 = add_texture_channel(img)
    img4 = (img4 - np.array(MEAN)) / np.array(STD)
    img4 = img4.transpose(2, 0, 1)[None, ...]  # (1,4,H,W)
    img_tensor = torch.tensor(img4, dtype=torch.float32, device=DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W)

    # probability overlay
    prob_vis = (prob * 255).astype(np.uint8)
    prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, prob_color, 0.4, 0)
    cv2.imwrite(str(SEG_DEBUG_DIR / f"{tile_id}_prob_overlay.jpg"), overlay)

    # binary mask
    mask = (prob >= prob_thresh).astype(np.uint8)

    # clean mask to separate touching crowns
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # slight erosion helps cut thin “bridges”
    # mask = cv2.erode(mask, kernel, iterations=1)

    # write mask layer to debug directory
    cv2.imwrite(str(SEG_DEBUG_DIR / f"{tile_id}_mask.png"), mask * 255)

    # distance transform + watershed
    if mask.sum() == 0:
        print(f"{tile_id}: empty mask after thresholding")
        return [], [], []

    dist = distance_transform_edt(mask)
    coords = peak_local_max(
        dist,
        min_distance=min_distance,
        threshold_rel=dt_thresh_rel,
        exclude_border=False
    )

    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i

    labels = watershed(-dist, markers, mask=mask)

    # watershed region statistics
    areas = []
    for rid in np.unique(labels):
        if rid == 0:
            continue
        ys, xs = np.where(labels == rid)
        areas.append(len(xs))

    tqdm.write(f"{tile_id}: regions={len(areas)}, "
        f"area median={np.median(areas):.1f}, "
        f"p90={np.percentile(areas, 90):.1f}, "
        f"p99={np.percentile(areas, 99):.1f}")

    # extract boxes & per-box scores
    H, W = mask.shape
    boxes = []
    scores = []
    nms_scores = []
    for rid in np.unique(labels):
        if rid == 0:
            continue

        region_mask = (labels == rid)
        area_px = int(region_mask.sum())
        if area_px < min_region_pixels or area_px > max_region_pixels:
            continue

        ys, xs = np.where(region_mask)
        cy = ys.mean()
        cx = xs.mean()

        # derive box size from area
        # approximate crown radius from area of a circle, then expand a bit
        radius = np.sqrt(area_px / np.pi)
        side = int(2 * radius * 1.2)   # 1.2 is a padding factor, tuneable

        # clamp side to a reasonable range in pixels
        side = max(8, min(side, 80))

        half = side / 2.0
        x1 = int(round(cx - half))
        x2 = int(round(cx + half))
        y1 = int(round(cy - half))
        y2 = int(round(cy + half))

        # clip to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W - 1, x2)
        y2 = min(H - 1, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append((x1, y1, x2, y2))

        # score from the region
        region_prob = prob[region_mask]
        mean_p = float(region_prob.mean())
        scores.append(mean_p)

        area_px = int(region_mask.sum())
        nms_score = mean_p * area_px ** 0.5   # second term big -> bigger boxes win out more
        nms_scores.append(nms_score)
        

    # No heuristic detection for base debugging
    # boxes = []
    # scores = []
    # for rid in np.unique(labels):
    #     if rid == 0:
    #         continue
    #     ys, xs = np.where(labels == rid)

    #     # --- very loose debug mode, no size/aspect filters ---
    #     if len(xs) < 5:
    #         continue  # only ignore truly tiny noise

    #     x1, x2 = xs.min(), xs.max()
    #     y1, y2 = ys.min(), ys.max()

    #     boxes.append((x1, y1, x2, y2))
    #     scores.append(float(prob[y1:y2, x1:x2].mean()))

    # draw mask watershed seeds and write to map
    vis_regions = img.copy()
    for rid in np.unique(labels):
        if rid == 0:
            continue
        ys, xs = np.where(labels == rid)
        cy, cx = int(ys.mean()), int(xs.mean())
        cv2.circle(vis_regions, (cx, cy), 2, (0, 0, 255), -1)
    cv2.imwrite(str(SEG_DEBUG_DIR / f"{tile_id}_region_centers.jpg"), vis_regions)


    # draw raw (pre-NMS) boxes
    vis = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(SEG_DEBUG_DIR / f"{tile_id}_boxes_raw.jpg"), vis)

    # run NMS below on (boxes, scores)
    tqdm.write(f"{tile_id}: {len(boxes)} instance boxes before NMS "
          f"(prob stats min={prob.min():.3f}, max={prob.max():.3f}, mean={prob.mean():.3f})")
    

    return boxes, scores, nms_scores


# Filter to try and remove ground boxes using laplacian filter like in patch network(doesn't work tmk)
def is_ground_like_box(img_bgr, x1, y1, x2, y2,
                       var_thresh=15.0, lap_thresh=5.0):
    # Crop
    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return True  # degenerate, discard

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Construct structure layer
    var = gray.var()
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_mean = np.abs(lap).mean()

    # Smooth, low-variance, low-Laplacian = likely ground
    return (var < var_thresh) and (lap_mean < lap_thresh)

def nms_boxes(boxes, scores, nms_scores, iou_thresh=0.5, contain_thresh=0.3):
    """
    NMS that also suppresses boxes when a large fraction of the smaller
    box is overlapped (containment)

    boxes: list of (x1, y1, x2, y2)
    scores: list of floats
    iou_thresh: standard IoU-based NMS threshold
    contain_thresh: intersection / min(area_i, area_j) threshold
    """
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    nms_scores = np.array(nms_scores, dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # half-open boxes: [x1, x2), [y1, y2)
    areas = (x2 - x1) * (y2 - y1)
    order = nms_scores.argsort()[::-1]  # highest score first

    keep_idxs = []
    while order.size > 0:
        i = order[0]
        keep_idxs.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # standard IoU
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        # overlap relative to smaller box
        smaller_areas = np.minimum(areas[i], areas[order[1:]]) + 1e-6
        overlap_small = inter / smaller_areas

        # suppress if either IoU high OR small-box overlap high
        suppress = (iou > iou_thresh) | (overlap_small > contain_thresh)

        # keep only those that are not suppressed
        inds = np.where(~suppress)[0]
        order = order[inds + 1]

    boxes_keep  = [tuple(map(int, boxes[i])) for i in keep_idxs]
    scores_keep = [float(scores[i]) for i in keep_idxs]
    return boxes_keep, scores_keep



# Image segmentation using mask layer and watershed
records = []
tifs = sorted(list(INPUT_DIR.glob("P*.tif")))
it = 0
print(f"{len(tifs)} images were found for segmentation/detection.")

for img_path in tqdm(tifs, desc="Inference on tiles"):
    tile_id = img_path.stem  # e.g., 'P0', 'P37'
    boxes_raw, scores_raw, scores_weighted = segment_and_boxes_on_tile(model, img_path, tile_id)

    # NMS (low IoU threshold = stricter overlap collapse)
    boxes_nms, scores_nms = nms_boxes(boxes_raw, scores_raw, scores_weighted, iou_thresh=0.2, contain_thresh=0.2)
    tqdm.write(f"{tile_id}: {len(boxes_nms)} boxes after NMS")

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    # save final boxes overlay
    vis = img.copy()
    for (x1, y1, x2, y2) in boxes_nms:
        if is_ground_like_box(img, x1, y1, x2, y2):
            continue  # skip ground-like region

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(SEG_DEBUG_DIR / f"{tile_id}_boxes_final.jpg"), vis)

    # write CSV rows, I can collapse this and above statement later
    for (x1, y1, x2, y2), score in zip(boxes_nms, scores_nms):
        records.append({
            "left":   int(x1),
            "bottom": int(h - y2),
            "right":  int(x2),
            "top":    int(h - y1),
            "score":  float(score),
            "label":  "Tree",
            "height": np.nan,
            "area":   int((x2 - x1) * (y2 - y1)),
            "Site":   "ABBY",
            "partition_id": tile_id,
        })
    
    # it += 1   # just if I want to run P0 tile
    # if it == 3:
    #     break

OUTPUT_CSV = DATA_DIR / "unet_seg_boxes.csv"
pd.DataFrame(records).to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(records)} boxes to {OUTPUT_CSV}")



end = timeit.default_timer()
print(f"Preprocess Time: {preprocess_time - start:.2f}")
print(f"Training Time: {train_time - start:.2f}")
print(f"Map Generation and Classification Time: {end - train_time:.2f}")
print(f"Time elapsed: {end - start:.2f}")


# cd ../../nfs/veritas/link55/testing/trees
# export CUDA_VISIBLE_DEVICES=1