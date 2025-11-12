import pandas as pd
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.transform import rowcol
from PIL import Image, ImageDraw
import os

# Paths
CSV_FILE = "a_trees_with_partitions.csv"       # ground truth labels
TIF_FOLDER = "tif_partitions/"        # from partition script
OUTPUT_FOLDER = "tif_with_boxes/"     # folder to save output images

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Read bounding boxes
df = pd.read_csv(CSV_FILE, usecols=["left", "bottom", "right", "top", "partition_id"], dtype={"partition_id": str})

# Loop over all partitions
for partition_id in df["partition_id"].unique():
    if partition_id == "-1":
        continue

    tif_path = os.path.join(TIF_FOLDER, f"{partition_id}.tif")
    if not os.path.exists(tif_path):
        print(f"Skipping {tif_path}, file not found")
        continue

    # Open the TIFF
    with rasterio.open(tif_path) as src:
        img_array = src.read()                  # (bands, height, width)
        img = reshape_as_image(img_array)       # (height, width, bands) or you can think as (rows, columns, bands)
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        # Filter boxes in this partition
        boxes = df[df["partition_id"] == partition_id]

        # Takes me back having to deal with a million different reference systems and trying to make some transformation work
        transform = src.transform
        height = pil_img.height

        # Draw each bounding box
        for _, row in boxes.iterrows():
            row_top, col_left = rowcol(transform, row["left"], row["top"])
            row_bottom, col_right = rowcol(transform, row["right"], row["bottom"])
            # Convert to pixel based coords
            x1, y1 = col_left, row_top
            x2, y2 = col_right, row_bottom
            
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

    # Save output
    out_path = os.path.join(OUTPUT_FOLDER, f"{partition_id}.tif")
    pil_img.save(out_path)
    print(f"Saved {out_path}")