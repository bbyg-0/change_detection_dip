"""Ekspor hasil Julia (npz) menjadi GeoTIFF untuk tiap kota."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from config import OUTPUT_DIR


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _transform(meta):
    gt = meta["geotransform"]
    return from_origin(gt["west"], gt["north"], gt["dx"], abs(gt["dy"]))


def export_city(slug):
    base = OUTPUT_DIR / "L1" / slug
    meta = json.loads((base / "prep" / "meta.json").read_text())
    transform = _transform(meta)
    h, w = meta["tile"]["h"], meta["tile"]["w"]
    written = []

    di_npz = base / "di" / f"{slug}_di.npz"
    if di_npz.exists():
        z = np.load(di_npz)
        arr = z["di"].astype(np.float32)
        mask = z["mask"].astype(bool)
        arr = np.where(mask, arr, np.nan)
        out = base / "di" / f"{slug}_di.tif"
        with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=1,
                           dtype="float32", crs="EPSG:4326", transform=transform,
                           nodata=np.nan, compress="deflate") as dst:
            dst.write(arr, 1)
        written.append(out)

    cmap_dir = base / "change_map"
    if cmap_dir.exists():
        for npz in sorted(cmap_dir.glob("*.npz")):
            z = np.load(npz)
            if "change" not in z.files:
                continue
            arr = z["change"].astype(np.uint8)
            out = cmap_dir / (npz.stem + ".tif")
            with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=1,
                               dtype="uint8", crs="EPSG:4326", transform=transform,
                               nodata=0, compress="deflate") as dst:
                dst.write(arr, 1)
            written.append(out)

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    a = ap.parse_args()
    if a.cities:
        slugs = [slugify(c) for c in a.cities]
    else:
        slugs = sorted(p.name for p in (OUTPUT_DIR / "L1").iterdir() if p.is_dir())
    for slug in slugs:
        files = export_city(slug)
        print(f"{slug}: {len(files)} file GeoTIFF")


if __name__ == "__main__":
    main()
