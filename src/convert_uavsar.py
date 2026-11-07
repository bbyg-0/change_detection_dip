"""Convert UAVSAR P-band GRD/INC/SLOPE binary files to GeoTIFF.

Requirements: pip install numpy rasterio
NOTE: this script has not been tested on real data. Verify the output
(width/height vs the .ann file, location vs the .kmz) before relying on it.
"""
import os
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

# ----------------------------------------------------------------------
# SETTINGS - edit these
# ----------------------------------------------------------------------
# Folder containing the dataset files (also used as the output folder).
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(ROOT / "data/T2")

# Filename pieces, e.g. tukhwy_01812_17057_014_170606_PL09043020_30HVVV_CX_01.grd
BASE = "eelriv_15807_23014_024_230913_PL09043020"  # everything before the resolution code
RES = "05"      # "30" = 3.0 arcsec (~90 m), "05" = 0.5 arcsec (~15 m)
CX_VV = "CX_01"  # crosstalk status + version number

OUT_DIR = DATA_DIR  # change if you want outputs somewhere else
# ----------------------------------------------------------------------


def path(folder, name):
    """Join folder and filename (works with or without a trailing slash)."""
    return os.path.join(folder, name)


def read_ann(ann_path):
    """Parse 'keyword (units) = value ; comment' lines into a dict."""
    params = {}
    with open(ann_path, errors="ignore") as f:
        for line in f:
            line = line.split(";")[0]
            m = re.match(r"\s*(\S+)\s*(?:\([^)]*\))?\s*=\s*(.+)", line)
            if m:
                params[m.group(1)] = m.group(2).strip()
    return params


def geometry(ann_path):
    """Return (rows, cols, affine transform) from the annotation file."""
    a = read_ann(ann_path)
    rows = int(float(a["grd_mag.set_rows"]))
    cols = int(float(a["grd_mag.set_cols"]))
    dx = abs(float(a["grd_mag.col_mult"]))
    dy = abs(float(a["grd_mag.row_mult"]))
    # row_addr / col_addr are the CENTER of the upper-left pixel;
    # GeoTIFF wants the upper-left CORNER, so shift by half a pixel.
    west = float(a["grd_mag.col_addr"]) - dx / 2
    north = float(a["grd_mag.row_addr"]) + dy / 2
    return rows, cols, from_origin(west, north, dx, dy)


def write_tif(out_path, bands, transform, descriptions=None):
    """Write a 2D (single band) or 3D (bands, rows, cols) array as GeoTIFF."""
    bands = np.asarray(bands)
    if bands.ndim == 2:
        bands = bands[None]
    is_float = bands.dtype.kind == "f"
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=bands.shape[1],
        width=bands.shape[2],
        count=bands.shape[0],
        dtype=bands.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan if is_float else None,
        compress="deflate",
    ) as dst:
        dst.write(bands)
        if descriptions:
            for i, d in enumerate(descriptions, 1):
                dst.set_band_description(i, d)
    print("wrote", out_path)


def main():
    ann = path(DATA_DIR, f"{BASE}_{RES}_{CX_VV}.ann")
    rows, cols, tf = geometry(ann)
    print(f"rows={rows}, cols={cols}")

    # 1) Power bands: HH, HV, VV stacked into one 3-band GeoTIFF
    pols = ["HHHH", "HVHV", "VVVV"]
    stack = []
    for p in pols:
        f = path(DATA_DIR, f"{BASE}_{RES}{p}_{CX_VV}.grd")
        a = np.fromfile(f, dtype="<f4").reshape(rows, cols)
        a[(a <= 0) | ~np.isfinite(a)] = np.nan  # mask invalid pixels (verify!)
        stack.append(a)
    write_tif(path(OUT_DIR, "sigma0_power.tif"), stack, tf, pols)

    # 2) Incidence angle (radians)
    inc = np.fromfile(
        path(DATA_DIR, f"{BASE}_{RES}_{CX_VV}.inc"), dtype="<f4"
    ).reshape(rows, cols)
    write_tif(path(OUT_DIR, "incidence.tif"), inc, tf, ["inc_rad"])

    # 3) Slope: two values per pixel (east, north)
    sl = np.fromfile(
        path(DATA_DIR, f"{BASE}_{RES}_{CX_VV}.slope"), dtype="<f4"
    ).reshape(rows, cols, 2)
    write_tif(
        path(OUT_DIR, "slope.tif"), np.moveaxis(sl, 2, 0), tf,
        ["slope_east", "slope_north"],
    )

    # 4) Optional: amplitude of a complex cross term (e.g. HHHV)
    # c = np.fromfile(
    #     path(DATA_DIR, f"{BASE}_{RES}HHHV_{CX_VV}.grd"), dtype="<c8"
    # ).reshape(rows, cols)
    # write_tif(path(OUT_DIR, "HHHV_amp.tif"), np.abs(c).astype("f4"), tf, ["HHHV_amp"])


if __name__ == "__main__":
    main()
