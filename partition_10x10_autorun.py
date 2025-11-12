import os
from pathlib import Path
import glob
import csv
import pandas as pd
import rasterio
from rasterio.windows import Window
import numpy as np

ROWS = 10
COLS = 10

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
tif_path = find_single("*.tif")
csv_path = find_single("*.csv")
# parent_dir = os.path.dirname
# out_dir = os.path.join(parent_dir, "output")

out_dir = "output"
os.makedirs(out_dir, exist_ok=True)
print(f"TIF: {tif_path}")
print(f"CSV: {csv_path}")
print(f"Files output dir: {out_dir}")

df = pd.read_csv(csv_path)

# Partition Tifs
with rasterio.open(tif_path) as tif:
    xmin, ymin, xmax, ymax = tif.bounds
    width, height = tif.width, tif.height
    print("Width, height:", width, height)
    partition_width = width // COLS    # should be 1000 x 1000
    partition_height = height // ROWS

    # Resolution for bbox logic later
    xres, yres = tif.res   # pixel size in m
    print(f"Pixel resolution: {xres} x {yres}")

    for row in range(ROWS):
        for column in range(COLS):
            pid = f"P{row * COLS + column}"
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


# Partition CSV
df = pd.read_csv(csv_path)
for col in ["left", "right", "top", "bottom"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Should be 1000 x 1000 given our dataset
dx = (xmax - xmin) / COLS
dy = (ymax - ymin) / ROWS

print("Grid difference:", dx, "x", dy)

partition_ids = []
for _, row in df.iterrows():
    left, bottom, right, top = row["left"], row["bottom"], row["right"], row["top"]
    # Determine which partitions the bbox is in
    left_p = int(np.floor((left - xmin) / dx))
    right_p = int(np.floor((right - xmin) / dx))
    bottom_p = int(np.floor((bottom - ymin) / dy))
    top_p = int(np.floor((top - ymin) / dy))

    # Clip to valid indices fior edge interaction
    left_p = np.clip(left_p, 0, COLS - 1)
    right_p = np.clip(right_p, 0, COLS - 1)
    bottom_p = np.clip(bottom_p, 0, ROWS - 1)
    top_p = np.clip(top_p, 0, ROWS - 1)
    if left_p != right_p or bottom_p != top_p:
        pid = "-1" 
    else:
        pid = f"P{int((ROWS - bottom_p - 1) * COLS + left_p)}"
    partition_ids.append(pid)

print("Number of bad trees", partition_ids.count("-1"))
print("Number of trees", len(partition_ids))

df["partition_id"] = partition_ids
df.to_csv(os.path.join(out_dir, "a_trees_with_partitions.csv"), index=False)


print("Finished Partioning")
