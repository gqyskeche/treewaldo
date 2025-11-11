#!/usr/bin/env python3
"""
Partition a raster (fixed 10x10) and assign CSV bounding boxes to partitions.

Place this script in the SAME folder as:
  - exactly one GeoTIFF (*.tif)
  - exactly one CSV (*.csv) with columns: left, bottom, right, top
    (Optionally a 'geo_index' column like '551000_5069000' which is lower-left (xmin,ymin).)

Run:
  python partition_10x10_autorun.py

Outputs (inside a new tiles folder):
  - <tif_basename>_tiles_10x10/
      ├─ P01.tif … P100.tif
      └─ <csv_basename>_with_partitions.csv

partition_id logic:
  - P01…P100 if the bbox centroid lies inside one grid cell AND the bbox does not
    touch/straddle a grid line (within tolerance).
  - -1 if bbox touches a grid line or lies outside image bounds.
"""

import os
import sys
import glob
import math
import re
import csv
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

# ---- Config ----
ROWS = 10
COLS = 10
BOUNDARY_TOL = 0.0   # meters. Set >0 to treat "near-line" as on-boundary.

# ---------- Helpers ----------
def find_single(pattern: str) -> str:
    files = [f for f in glob.glob(pattern) if os.path.isfile(f)]
    if len(files) == 0:
        print(f"ERROR: No files match {pattern!r} in {os.getcwd()}")
        sys.exit(1)
    if len(files) > 1:
        print(f"ERROR: Multiple files match {pattern!r}: {files}. Keep only one.")
        sys.exit(1)
    return files[0]

def parse_neon_lower_left_from_tif_name(tif_path):
    """
    Parse NEON-style filename endings (LOWER-LEFT, xmin,ymin):
      ..._<EASTING6>_<NORTHING7>_image.tif   OR   ..._<EASTING6>_<NORTHING7>.tif
    Returns (xmin, ymin) if found else None.
    """
    fname = os.path.basename(tif_path)
    m = re.search(r'_(\d{6})_(\d{7})(?:_image)?\.tif$', fname)
    if not m:
        return None
    xmin = float(m.group(1))
    ymin = float(m.group(2))
    return xmin, ymin

def parse_lower_left_from_geo_index(df):
    """
    If CSV has a 'geo_index' like '551000_5069000', treat it as (xmin,ymin).
    Uses the most common value if there are multiple.
    """
    if "geo_index" not in df.columns:
        return None
    # take mode of geo_index to avoid mixed tiles
    val = df["geo_index"].dropna().astype(str).mode()
    if val.empty:
        return None
    s = val.iloc[0]
    m = re.match(r'^\s*(\d{6})_(\d{7})\s*$', s)
    if not m:
        return None
    xmin = float(m.group(1))
    ymin = float(m.group(2))
    return xmin, ymin

def compute_grid(xmin, ymin, xmax, ymax):
    dx = (xmax - xmin) / COLS
    dy = (ymax - ymin) / ROWS
    verts = [xmin + k*dx for k in range(1, COLS)]
    hors  = [ymin + k*dy for k in range(1, ROWS)]
    return dx, dy, verts, hors

def bbox_touches_boundary(L, B, R, T, verticals, horizontals, tol=BOUNDARY_TOL):
    for v in verticals:   # x = v
        if (L < v < R) or math.isclose(L, v, abs_tol=tol) or math.isclose(R, v, abs_tol=tol):
            return True
    for h in horizontals: # y = h
        if (B < h < T) or math.isclose(B, h, abs_tol=tol) or math.isclose(T, h, abs_tol=tol):
            return True
    return False

def centroid_to_pid(cx, cy, xmin, ymin, dx, dy):
    col = int((cx - xmin) // dx)
    row = int((cy - ymin) // dy)
    col = max(0, min(COLS - 1, col))
    row = max(0, min(ROWS - 1, row))
    return f"P{row * COLS + col + 1:02d}"

def export_tiles_pixel(ds, out_dir):
    """Export 10x10 tiles using pixel windows (works even without geotransform)."""
    os.makedirs(out_dir, exist_ok=True)
    meta = ds.meta.copy()
    H, W = ds.height, ds.width
    dh = H / ROWS
    dw = W / COLS

    for r in tqdm(range(ROWS), desc="Exporting TIFF rows", ncols=80):
        for c in tqdm(range(COLS), desc=f"row {r+1:02d}", leave=False, ncols=80):
            pid = f"P{(r * COLS + c + 1):02d}"
            row0 = int(round(r * dh)); row1 = int(round((r + 1) * dh))
            col0 = int(round(c * dw)); col1 = int(round((c + 1) * dw))
            window = Window(col_off=col0, row_off=row0, width=col1-col0, height=row1-row0)
            data = ds.read(window=window)
            meta_tile = meta.copy()
            try:
                new_transform = rasterio.windows.transform(window, ds.transform)
                meta_tile.update({"transform": new_transform})
            except Exception:
                pass
            meta_tile.update({"height": data.shape[1], "width": data.shape[2]})
            out_path = os.path.join(out_dir, f"{pid}.tif")
            with rasterio.open(out_path, "w", **meta_tile) as dst:
                dst.write(data)

def save_csv(df, out_csv_path):
    # Excel-friendly CSV (BOM + CRLF)
    df.to_csv(out_csv_path, index=False, encoding="utf-8-sig", lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    print(f"CSV with partitions: {out_csv_path}")

# ---------- Main ----------
def main():
    # Locate inputs
    tif_path = find_single("*.tif")
    csv_path = find_single("*.csv")

    tif_base = os.path.splitext(os.path.basename(tif_path))[0]
    csv_base = os.path.splitext(os.path.basename(csv_path))[0]
    out_tiles_dir = f"{tif_base}_tiles_10x10"
    os.makedirs(out_tiles_dir, exist_ok=True)

    print(f"Using TIF: {tif_path}")
    print(f"Using CSV: {csv_path}")
    print(f"Tiles output dir: {out_tiles_dir}")

    # Read CSV first (we may use geo_index fallback)
    df = pd.read_csv(csv_path)

    # Open raster & determine bounds (prefer real transform, else lower-left from TIF name, else geo_index)
    with rasterio.open(tif_path) as ds:
        if (ds.transform is not None) and (ds.transform != Affine.identity()):
            xmin, ymin, xmax, ymax = ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top
            print("Geospatial transform found in raster.")
            map_ready = True
        else:
            print("No valid geotransform; trying to reconstruct LOWER-LEFT from filename...")
            ll = parse_neon_lower_left_from_tif_name(tif_path)
            if ll is None:
                print("No luck from filename; trying geo_index in CSV...")
                ll = parse_lower_left_from_geo_index(df)
            if ll is not None:
                xmin, ymin = ll
                xmax = xmin + 1000.0
                ymax = ymin + 1000.0
                map_ready = True
                print(f"Reconstructed LOWER-LEFT bounds: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
            else:
                map_ready = False
                print("Could not reconstruct map bounds. Will export tiles, but partition_id will be -1 for all rows.")

        # Export tiles
        export_tiles_pixel(ds, out_tiles_dir)

    # Ensure required columns exist & numeric
    required_cols = {"left", "bottom", "right", "top"}
    if not required_cols.issubset(df.columns):
        print(f"ERROR: CSV must contain columns: {sorted(required_cols)}")
        sys.exit(1)
    for col in ["left", "bottom", "right", "top"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Assign partition_id
    out_csv_path = os.path.join(out_tiles_dir, f"{csv_base}_with_partitions.csv")
    partition_ids = []
    if map_ready:
        dx, dy, verticals, horizontals = compute_grid(xmin, ymin, xmax, ymax)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV rows", ncols=80):
            L = row["left"]; B = row["bottom"]; R = row["right"]; T = row["top"]
            if pd.isna(L) or pd.isna(B) or pd.isna(R) or pd.isna(T):
                partition_ids.append(-1); continue
            if bbox_touches_boundary(L, B, R, T, verticals, horizontals, tol=BOUNDARY_TOL):
                partition_ids.append(-1); continue
            cx = (L + R) / 2.0; cy = (B + T) / 2.0
            if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
                partition_ids.append(-1); continue
            partition_ids.append(centroid_to_pid(cx, cy, xmin, ymin, dx, dy))
    else:
        partition_ids = [-1] * len(df)

    df["partition_id"] = partition_ids

    # Save CSV (inside tiles folder)
    save_csv(df, out_csv_path)

    print("Done.")
    print(f"Tiles folder: {out_tiles_dir}")
    print(f"CSV with partitions: {out_csv_path}")

if __name__ == "__main__":
    main()
