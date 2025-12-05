import pandas as pd
import numpy as np
import timeit
import re
from collections import defaultdict
import matplotlib.pyplot as plt

REF_CSV = "a_trees_with_partitions_multi_image.csv"  # reference bounding boxes in meters
# cnn_boxes | baseline_boxes
PRED_CSV = "cnn_boxes.csv"  # predicted bounding boxes(most likely in pixel units top left is 0)
OVERLAP_THRESHOLD = 0.5  # IoR threshold
missing_origin_warned = set()

def extract_origin(geo_index_str):
    if pd.isna(geo_index_str):
        return None, None
    s = str(geo_index_str)
    m = re.search(r"(\d+)_(\d+)", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None

def save_reduced_confusion_matrix(TP, FP, FN, out_path="reduced_confusion_matrix_fancy.png"):
    data = np.array([
        [TP, FN],
        [FP, np.nan]
    ], dtype=float)

    row_labels = ["Actual Tree", "No GT Tree"]
    col_labels = ["Predicted Tree", "No Prediction"]

    fig, ax = plt.subplots(figsize=(6, 4))

    cmap = plt.cm.Blues
    masked_data = np.ma.masked_invalid(data)
    heatmap = ax.imshow(masked_data, cmap=cmap)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = "N/A" if np.isnan(val) else f"{int(val)}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="black")

    plt.title("Reduced Confusion Matrix", fontsize=14, pad=16)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()

def bbox_iou(ref, pred):
    """Intersection over reference area"""
    xA = max(ref[0], pred[0])
    yA = max(ref[1], pred[1])
    xB = min(ref[2], pred[2])
    yB = min(ref[3], pred[3])
    
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    
    area_ref = (ref[2] - ref[0]) * (ref[3] - ref[1])
    if area_ref == 0:
        return 0
    return inter_area / area_ref

ref_df = pd.read_csv(REF_CSV, usecols=["left", "bottom", "right", "top", "partition_id", "geo_index"], 
                     dtype={"left": float, "bottom": float, "right": float, "top": float, "partition_id": str, "geo_index": str})
pred_df = pd.read_csv(PRED_CSV, usecols=["left", "bottom", "right", "top", "partition_id"], 
                      dtype={"left": float, "bottom": float, "right": float, "top": float, "partition_id": str})

partition_origin = {}

for pid, group in ref_df.groupby("partition_id"):
    geo_idx = group["geo_index"].iloc[0]
    ox, oy = extract_origin(geo_idx)
    if ox is not None and oy is not None:
        partition_origin[pid] = (ox, oy)
        print(f"Partition {pid} origin: ({ox}, {oy})")
        

# optional: choose a fallback origin if something is missing back to original ABBY image
DEFAULT_ORIGIN_X = 551000
DEFAULT_ORIGIN_Y = 5069000

def pixel_to_utm(row):
    pid_str = str(row["partition_id"])

    # Optional: skip invalid sentinel partitions
    if pid_str == "-1":
        return row

    origin = partition_origin.get(pid_str)

    # If this partition has no GT trees, it won't be in partition_origin.
    # We don't need UTM coords for IoR (there is no ref box to compare to),
    # so just leave the box as-is (or only scale if you want).
    if origin is None:
        if pid_str not in missing_origin_warned:
            print(f"[WARN] No origin for partition_id={pid_str}; "
                  "leaving coords in pixel space (used only for FP counting).")
            missing_origin_warned.add(pid_str)
        return row   # <- no transform needed for evaluation

    origin_x, origin_y = origin

    # Numeric index of the partition, assuming format "p123"
    pid_num = int(pid_str[1:])

    # Local index (0–99) inside a 10x10 grid, resets every 100 partitions
    local = pid_num % 100
    col   = local % 10
    row_i = local // 10

    x_off = 100 * col
    y_off = 900 - 100 * row_i

    scale = 0.1  # pixel -> meters

    row["left"]   = row["left"]  * scale + origin_x + x_off
    row["right"]  = row["right"] * scale + origin_x + x_off
    row["bottom"] = row["bottom"]* scale + origin_y + y_off
    row["top"]    = row["top"]   * scale + origin_y + y_off

    return row

start = timeit.default_timer()

pred_df = pred_df.apply(pixel_to_utm, axis=1)

print("Number of partitions with origin:", len(partition_origin))
print("Example origins:", list(partition_origin.items())[:5])
# print("first 10 predictions", pred_df.head(10))

TP = 0
FP = 0
FN = 0
total_iou_tp = 0.0
iou_tp_values = []
i = 0   # Iterations

partition_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

pred_matched = {pid: set() for pid in pred_df["partition_id"].unique()} # Mark predicted trees used for precision(subtract set)
for _, ref in ref_df.iterrows():
    ref_box = [ref.left, ref.bottom, ref.right, ref.top]
    pid = ref["partition_id"]
    
    if i == 275860:
        print("Past first batch")
        break
    if pid == "-1":
        i += 1
        continue
    if i % 500 == 0: 
        print("Partition ID:", pid)
        print("Iteration:", i)
        print("Current TP, FP, FN:", TP, FP, FN)
    
    i += 1

    # Only check predictions in the same partition
    preds_in_pid = pred_df[pred_df["partition_id"] == pid]

    best_iou = 0.0
    best_pred_idx = None

    for pred_idx, pred in preds_in_pid.iterrows():
        pred_box = [pred.left, pred.bottom, pred.right, pred.top]
        iou = bbox_iou(ref_box, pred_box)

        if iou > best_iou:
            best_iou = iou
            best_pred_idx = pred_idx

    # Count TP / FN and mark exactly ONE prediction as matched
    if best_iou >= OVERLAP_THRESHOLD and best_pred_idx is not None:
        TP += 1
        total_iou_tp += best_iou              # track IoR for TPs
        iou_tp_values.append(best_iou)
        partition_stats[pid]["tp"] += 1       # per-partition TP
        pred_matched[pid].add(best_pred_idx)
    else:
        FN += 1                               # no BB corresponding to this reference
        partition_stats[pid]["fn"] += 1       # Since there no BB corresponding to reference BBox

# Determine how many predicted trees were not used(FP)
for pid, group in pred_df.groupby("partition_id"):
    for pred_idx in group.index:
        if pred_idx not in pred_matched[pid]:
            FP += 1
            partition_stats[pid]["fp"] += 1

recall = TP / (TP+FN)       if TP + FN > 0 else 0
precision = TP / (TP + FP)  if TP + FP > 0 else 0
f1_score = 2 * (recall * precision) / (recall + precision)  if (recall + precision) > 0 else 0
avg_iou_tp = total_iou_tp / TP if TP > 0 else 0.0
if iou_tp_values:
    high_quality = sum(i >= 0.75 for i in iou_tp_values) / len(iou_tp_values)
else:
    high_quality = 0.0

print(f"High-quality TPs (IoR ≥ 0.75): {high_quality*100:.2f}%")
print(f"TP: {TP} | FP: {FP} | FN: {FN}")
save_reduced_confusion_matrix(TP, FP, FN)
print(f"Detected {TP}/{TP+FP} prediction trees correctly ({precision*100:.2f}% Precision)")
print(f"Detected {TP}/{TP+FN} reference trees correctly ({recall*100:.2f}% Recall)")
print(f"F1 Score: {f1_score}")
print(f"Average IoR over TPs: {avg_iou_tp:.4f}")

rows = []
for pid, stats in partition_stats.items():
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    n_ref = tp + fn
    n_pred = tp + fp
    prec_pid = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec_pid = tp / (tp + fn) if tp + fn > 0 else 0.0
    rows.append({
        "partition_id": pid,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_ref": n_ref,
        "n_pred": n_pred,
        "precision": prec_pid,
        "recall": rec_pid,
    })

if rows:
    part_df = pd.DataFrame(rows)
    part_df.to_csv("partition_eval_summary.csv", index=False)
    print("Saved per-partition summary to partition_eval_summary.csv")
if iou_tp_values:
    plt.figure(figsize=(6, 4))
    plt.hist(iou_tp_values, bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], edgecolor="black")
    plt.xlabel("IoR of True Positives")
    plt.ylabel("Count")
    plt.title("Distribution of IoR for True Positive Detections")
    plt.tight_layout()
    plt.savefig("tp_iou_histogram.png", dpi=220)
    plt.close()
    print("Saved IoR histogram to tp_iou_histogram.png")

n_ref_list = []
recall_list = []

for pid, stats in partition_stats.items():
    tp = stats["tp"]
    fn = stats["fn"]
    n_ref = tp + fn

    if n_ref == 0:
        continue

    recall = tp / (tp + fn)
    n_ref_list.append(n_ref)
    recall_list.append(recall)

if len(n_ref_list) > 1:
    pearson_corr = np.corrcoef(n_ref_list, recall_list)[0, 1]
else:
    pearson_corr = 0.0

print(f"Pearson correlation (density vs recall): {pearson_corr:.4f}")

if len(n_ref_list) > 1:
    x = np.array(n_ref_list, dtype=float)
    y = np.array(recall_list, dtype=float)

    # Fit a line y = m*x + b
    m, b = np.polyfit(x, y, 1)

    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = m * x_line + b

    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.6, label="Partitions")
    plt.plot(x_line, y_line, label="Best fit", linewidth=2)

    plt.xlabel("Number of ground-truth trees in partition (n_ref)")
    plt.ylabel("Recall per partition")
    plt.title(f"Density vs Recall (Pearson r = {pearson_corr:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("density_vs_recall_scatter.png", dpi=220)
    plt.close()
    print("Saved: density_vs_recall_scatter.png")

end = timeit.default_timer()
print("Elasped time:", end - start)