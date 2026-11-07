import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

def read_ann(path):
    p = {}
    for line in open(path, errors="ignore"):
        line = line.split(";")[0]
        m = re.match(r"\s*(\S+)\s*(?:\([^)]*\))?\s*=\s*(.+)", line)
        if m:
            p[m.group(1)] = m.group(2).strip()
    return p

def geometry(ann):
    a = read_ann(ann)
    rows = int(float(a["grd_mag.set_rows"]))
    cols = int(float(a["grd_mag.set_cols"]))
    dx = abs(float(a["grd_mag.col_mult"]))
    dy = abs(float(a["grd_mag.row_mult"]))
    # row_addr / col_addr = CENTER of the upper-left pixel -> shift to its corner
    west  = float(a["grd_mag.col_addr"]) - dx / 2
    north = float(a["grd_mag.row_addr"]) + dy / 2
    return rows, cols, from_origin(west, north, dx, dy)

def write_tif(out, bands, transform, descriptions=None):
    bands = np.asarray(bands)
    if bands.ndim == 2:
        bands = bands[None]
    with rasterio.open(out, "w", driver="GTiff", height=bands.shape[1],
                       width=bands.shape[2], count=bands.shape[0],
                       dtype=bands.dtype, crs="EPSG:4326",
                       transform=transform, nodata=np.nan if bands.dtype.kind == "f" else None,
                       compress="deflate") as dst:
        dst.write(bands)
        if descriptions:
            for i, d in enumerate(descriptions, 1):
                dst.set_band_description(i, d)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/T1"

ann = str(DATA / "eelriv_15807_20037_012_201113_PL09043020_05_CX_02.ann")
rows, cols, tf = geometry(ann)

# 1) Power bands: HH, HV, VV stacked into one 3-band GeoTIFF
pols = ["HHHH", "HVHV", "VVVV"]
stack = []
base = "eelriv_15807_20037_012_201113_PL09043020_05"
for p in pols:
    f = str(DATA / f"{base}{p}_CX_02.grd")
    a = np.fromfile(f, dtype="<f4").reshape(rows, cols)
    a[(a <= 0) | ~np.isfinite(a)] = np.nan   # mask no-data / invalid
    stack.append(a)
write_tif(str(DATA / "sigma0_power.tif"), stack, tf, pols)

# 2) Incidence angle (radians, float32)
inc = np.fromfile(str(DATA / f"{base}_CX_02.inc"), dtype="<f4").reshape(rows, cols)
write_tif(str(DATA / "incidence.tif"), inc, tf, ["inc_rad"])

# 3) Slope: 2 values per pixel (east, north)
sl = np.fromfile(str(DATA / f"{base}_CX_02.slope"), dtype="<f4").reshape(rows, cols, 2)
write_tif(str(DATA / "slope.tif"), np.moveaxis(sl, 2, 0), tf, ["slope_east", "slope_north"])

# 4) Optional: complex cross-term as amplitude (e.g. HHHV)
# c = np.fromfile(str(DATA / f"{base}HHHV_CX_02.grd"), dtype="<c8").reshape(rows, cols)
# write_tif(str(DATA / "HHHV_amp.tif"), np.abs(c).astype("f4"), tf, ["HHHV_amp"])
