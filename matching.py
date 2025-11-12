import pandas as pd
import timeit

REF_CSV = "a_trees_with_partitions.csv"  # reference bounding boxes in meters
PRED_CSV = "baseline_boxes.csv"  # predicted bounding boxes (same units)
OVERLAP_THRESHOLD = 0.3  # IoR threshold

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
print("first 10 predictions", pred_df.head(10))

correct = 0
i = 0
valid_count = 0
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
    valid_count += 1

    # Only check predictions in the same partition
    preds_in_pid = pred_df[pred_df["partition_id"] == pid]
    
    matched = False
    
    for _, pred in preds_in_pid.iterrows():
        pred_box = [pred.left, pred.bottom, pred.right, pred.top]
        if bbox_iou(ref_box, pred_box) >= OVERLAP_THRESHOLD:
            matched = True
            break
    if matched:
        correct += 1

accuracy = correct / valid_count
print(f"Detected {correct}/{valid_count} trees correctly ({accuracy*100:.2f}% accuracy)")

end = timeit.default_timer()
print("Elasped time:", end - start)