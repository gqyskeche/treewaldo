import pandas as pd
import timeit

REF_CSV = "a_trees_with_partitions.csv"  # reference bounding boxes in meters
# cnn_boxes | baseline_boxes
PRED_CSV = "cnn_boxes.csv"  # predicted bounding boxes(most likely in pixel units top left is 0)
OVERLAP_THRESHOLD = 0.5  # IoR threshold

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

ref_df = pd.read_csv(REF_CSV, usecols=["left", "bottom", "right", "top", "partition_id"], 
                     dtype={"left": float, "bottom": float, "right": float, "top": float, "partition_id": str})
pred_df = pd.read_csv(PRED_CSV, usecols=["left", "bottom", "right", "top", "partition_id"], 
                      dtype={"left": float, "bottom": float, "right": float, "top": float, "partition_id": str})

def pixel_to_utm(row):
    # Manual conversion, can add auto detect resolution later
    pid = int(row['partition_id'][1:])
    x_off = 100 * (pid % 10)
    y_off = 900 - 100 * (pid // 10)
    
    partition_xmin = 551000
    partition_ymin = 5069000
    row["left"]   = row["left"] * 0.1 + partition_xmin + x_off
    row["right"]  = row["right"] * 0.1 + partition_xmin + x_off
    row["bottom"] = row["bottom"] * 0.1 + partition_ymin + y_off
    row["top"]    = row["top"] * 0.1 + partition_ymin + y_off
    return row

start = timeit.default_timer()

pred_df = pred_df.apply(pixel_to_utm, axis=1)
# print("first 10 predictions", pred_df.head(10))

TP = 0
FP = 0
FN = 0
i = 0   # Iterations
pred_matched = {pid: set() for pid in pred_df["partition_id"].unique()} # Mark predicted trees used for precision(subtract set)
for _, ref in ref_df.iterrows():
    ref_box = [ref.left, ref.bottom, ref.right, ref.top]
    pid = ref["partition_id"]
    
    if i == 31501:
        print("Past first batch")
        break
    if pid == "-1":
        i += 1
        continue
    if i % 500 == 0: 
        print("Partition ID:", pid)
        print("Iteration:", i)
    
    i += 1

    # Only check predictions in the same partition
    preds_in_pid = pred_df[pred_df["partition_id"] == pid]
    
    best_iou = 0
    
    for pred_idx, pred in preds_in_pid.iterrows():
        pred_box = [pred.left, pred.bottom, pred.right, pred.top]
        iou = bbox_iou(ref_box, pred_box)
        
        if iou > best_iou:
            best_iou = iou
            pred_matched[pid].add(pred_idx)

    if best_iou >= OVERLAP_THRESHOLD:
        TP += 1
    else:
        FN += 1 # Since there no BB corresponding to reference BBox

# Determine how many predicted trees were not used(FP)
for pid, group in pred_df.groupby("partition_id"):
    for pred_idx in group.index:
        if pred_idx not in pred_matched[pid]:
            FP += 1

recall = TP / (TP+FN)       if TP + FN > 0 else 0
precision = TP / (TP + FP)  if TP + FP > 0 else 0
f1_score = 2 * (recall * precision) / (recall + precision)  if (recall + precision) > 0 else 0
print(f"TP: {TP} | FP: {FP} | FN: {FN}")
print(f"Detected {TP}/{TP+FP} prediction trees correctly ({precision*100:.2f}% Precision)")
print(f"Detected {TP}/{TP+FN} reference trees correctly ({recall*100:.2f}% Recall)")
print(f"F1 Score: {f1_score}")

end = timeit.default_timer()
print("Elasped time:", end - start)