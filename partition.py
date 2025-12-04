import os
from pathlib import Path
import glob
import csv
import pandas as pd
import rasterio
from rasterio.windows import Window
import numpy as np
import timeit

ROWS = 10
COLS = 10
PARTITIONS_PER_IMAGE = ROWS * COLS

TARGET_GEO_INDICES = [
    "551000_5069000",
    "553000_5064000",
    "553000_5062000",
    "551000_5065000",
    "554000_5068000",
    "554000_5063000",
    "551000_5071000",
    "552000_5063000",
    "551000_5062000",
    "554000_5072000",
]

# Regex to auto detect csv and tif because lazy
def find_single(pattern: str):
    files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    if len(files) == 0:
       raise Exception(f"ERROR: Missing {pattern} file in {os.getcwd()}")
    if len(files) > 1:
        raise Exception(f"ERROR: Multiple files match {pattern}: {files}")
    return files[0]

# Main Program
# TODO: In the future make it so I don't have to be in the same dir, obtain path
# tif_path = find_single("*.tif")
csv_path = find_single("*.csv")
# parent_dir = os.path.dirname
# out_dir = os.path.join(parent_dir, "output")

out_dir = "partition_output"
os.makedirs(out_dir, exist_ok=True)
# print(f"TIF: {tif_path}")
print(f"CSV: {csv_path}")
print(f"Files output dir: {out_dir}")


start = timeit.default_timer()

df = pd.read_csv(csv_path)

# Clean numeric columns
for col in ["left", "right", "top", "bottom"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Default partition_id = "-1" for everything
df["partition_id"] = "-1"

# Partition Tifs
image_meta = {}

for img_idx, geo in enumerate(TARGET_GEO_INDICES):
    
    print(f"Processing image {geo}")
    # Try to find the TIF for this tile.
    tif_matches = glob.glob(f"*{geo}*.tif")

    if len(tif_matches) == 0:
        print(f"WARNING: No TIF found for geo_index {geo}, skipping.")
        continue
    elif len(tif_matches) > 1:
        raise Exception(f"ERROR: Multiple TIFs match geo_index {geo}: {tif_matches}")

    tif_path = tif_matches[0]
    print(f"Processing image {img_idx} with geo_index {geo}: {tif_path}")

    # Base PID offset: image 0 → [0..99], image 1 → [100..199], etc.
    base_pid = img_idx * PARTITIONS_PER_IMAGE

    with rasterio.open(tif_path) as tif:
        xmin, ymin, xmax, ymax = tif.bounds
        width, height = tif.width, tif.height
        print("Width, height:", width, height)
        partition_width = width // COLS    # should be 1000 x 1000
        partition_height = height // ROWS

        # Resolution for bbox logic later
        xres, yres = tif.res   # pixel size in m
        print(f"Pixel resolution: {xres} x {yres}")

        # World-space grid spacing
        dx = (xmax - xmin) / COLS
        dy = (ymax - ymin) / ROWS
        print(f"  Grid difference: {dx} x {dy}")

        # Store metadata for CSV logic
        image_meta[geo] = {
            "tif_path": tif_path,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "dx": dx,
            "dy": dy,
            "base_pid": base_pid,
        }

        for row in range(ROWS):
            for column in range(COLS):
                local_partition_index = row * COLS + column
                global_partition_index = base_pid + local_partition_index
                pid = f"P{global_partition_index}"

                window = Window(
                    col_off = column * partition_width,
                    row_off = row * partition_height,
                    width = partition_width,
                    height = partition_height,
                )

                transform = rasterio.windows.transform(window, tif.transform)
                profile = tif.profile # CRS should carry over
                profile.update({
                    "transform": transform,
                    "width": partition_width,
                    "height": partition_height,

                })

                out_path = os.path.join(out_dir, f"{pid}.tif")
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(tif.read(window=window))

print("Finished writing partitioned TIFs")

bad_trees = 0
total_trees = len(df)
total_processing_trees = 0

for idx, row in df.iterrows():
    geo_index = str(row.get("geo_index", ""))
    # Skip if this tree is not from one of the 10 target images
    if geo_index not in image_meta:
        # partition_id stays "-1"
        continue

    meta = image_meta[geo_index]
    xmin = meta["xmin"]
    ymin = meta["ymin"]
    dx = meta["dx"]
    dy = meta["dy"]
    base_pid = meta["base_pid"]

    left, bottom, right, top = row["left"], row["bottom"], row["right"], row["top"]
    total_processing_trees += 1

    # If any coord is NaN, mark as bad
    if np.isnan(left) or np.isnan(bottom) or np.isnan(right) or np.isnan(top):
        bad_trees += 1
        # partition_id stays "-1"
        continue

    # Determine which partitions the bbox is in
    left_p = int(np.floor((left - xmin) / dx))
    right_p = int(np.floor((right - xmin) / dx))
    bottom_p = int(np.floor((bottom - ymin) / dy))
    top_p = int(np.floor((top - ymin) / dy))

    # Clip to valid indices 
    left_p = np.clip(left_p, 0, COLS - 1)
    right_p = np.clip(right_p, 0, COLS - 1)
    bottom_p = np.clip(bottom_p, 0, ROWS - 1)
    top_p = np.clip(top_p, 0, ROWS - 1)

    # Overlap detection
    if left_p != right_p or bottom_p != top_p:
        pid = "-1" # technically automarked but eh
        bad_trees += 1
        continue
    
    # Compute local partition index (same logic as before, flipped in Y)
    local_partition_index = int((ROWS - bottom_p - 1) * COLS + left_p)

    # Add global offset for this image
    global_partition_index = base_pid + local_partition_index
    pid = f"P{global_partition_index}"

    df.at[idx, "partition_id"] = pid

    if idx % 1000 == 0:
        print(f"Iteration: {idx}, Geo Index: {geo_index}")


print("Number of trees with overlaps / invalid coords (partition_id = -1):", bad_trees)
print("Total number of trees:", total_trees)

# Save new CSV
output_csv = os.path.join(out_dir, "a_trees_with_partitions_multi_image.csv")
df.to_csv(output_csv, index=False)

print("Finished partitioning CSV.")
print(f"Output CSV: {output_csv}")

print("Finished Partioning")
print(f"Time Elapsed: {timeit.default_timer()- start}")