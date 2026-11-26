"""Ambil patch optik (NAIP & Sentinel-2) via Planetary Computer STAC.

Patch 128x128 di sekitar tiap titik sampel, dengan crosshair di tengah.
Gagal mengambil satu set tidak menghentikan proses; labeling tetap bisa
memakai SAR saja.
"""

import argparse
import json
import os
import time
from itertools import islice
from pathlib import Path

os.environ.setdefault("GDAL_HTTP_TIMEOUT", "30")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "2")

import numpy as np
import planetary_computer as pc
import rasterio
import shapely.geometry as geom
from PIL import Image
from pystac_client import Client
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform

from config import CITIES, PATCH_SIZE
from preprocess import city_slug, output_tile_dir

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

SETS = {
    "NAIP_2020": dict(collection="naip", datetime="2020-01-01/2020-12-31",
                      asset="image"),
    "NAIP_2022": dict(collection="naip", datetime="2022-01-01/2022-12-31",
                      asset="image"),
    "S2_2020": dict(collection="sentinel-2-l2a", datetime="2020-11-01/2020-11-30",
                    asset="visual", cloud=30, limit=3),
    "S2_2023": dict(collection="sentinel-2-l2a", datetime="2023-09-01/2023-09-30",
                    asset="visual", cloud=30, limit=3),
}


def _search(catalog, collection, bbox, dt, cloud=None, limit=10, retries=3):
    query = {"eo:cloud_cover": {"lt": cloud}} if cloud else None
    sortby = ([{"field": "properties.eo:cloud_cover", "direction": "asc"}]
              if cloud else None)
    last = None
    for attempt in range(retries):
        try:
            s = catalog.search(collections=[collection], bbox=bbox, datetime=dt,
                               query=query, sortby=sortby, limit=limit)
            return list(islice(s.items(), limit))
        except Exception as e:  # timeout/server
            last = e
            time.sleep(2 ** attempt)
    print(f"    ! search gagal ({collection} {dt}): {last}")
    return []


def _city_bbox(meta):
    gt = meta["geotransform"]
    h, w = meta["tile"]["h"], meta["tile"]["w"]
    west, north = gt["west"], gt["north"]
    east = west + w * gt["dx"]
    south = north + h * gt["dy"]
    return [min(west, east), min(north, south), max(west, east), max(north, south)]


def _open_items(items, asset):
    """Buka dataset COG ter-sign; kembalikan list (id, dataset)."""
    out = []
    for it in items:
        try:
            signed = pc.sign(it)
            if asset not in signed.assets:
                continue
            ds = rasterio.open(signed.assets[asset].href)
            out.append((it.id, ds))
        except Exception as e:
            print(f"    ! buka item gagal {it.id}: {str(e)[:80]}")
    return out


def _read_rgb(ds, lon, lat, patch):
    try:
        x, y = warp_transform("EPSG:4326", ds.crs, [lon], [lat])
        row, col = ds.index(x[0], y[0])
        win = Window(col - patch / 2, row - patch / 2, patch, patch)
        arr = ds.read(window=win, boundless=True, fill_value=0)
    except Exception:
        return None
    if arr.shape[0] < 3:
        return None
    rgb = np.transpose(arr[:3], (1, 2, 0))
    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.float64)
        lo, hi = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = (rgb * 255).astype(np.uint8)
    if rgb.max() == 0:
        return None
    return rgb


def fetch_city(city, sets=None, patch=PATCH_SIZE, limit=None):
    sets = sets or list(SETS)
    base = output_tile_dir(city)
    meta = json.loads((base / "prep" / "meta.json").read_text())
    with open(base / "samples" / "samples.csv") as f:
        import csv
        samples = list(csv.DictReader(f))
    if limit:
        samples = samples[:limit]

    out_dir = base / "patches"
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox = _city_bbox(meta)
    catalog = Client.open(CATALOG_URL)
    status = {}

    for key in sets:
        cfg = SETS[key]
        print(f"  {city} / {key}: search ...")
        items = _search(catalog, cfg["collection"], bbox, cfg["datetime"],
                        cloud=cfg.get("cloud"), limit=cfg.get("limit", 10))
        datasets = _open_items(items, cfg["asset"])
        print(f"    {len(datasets)} item dibuka")
        ok = 0
        for s in samples:
            lon, lat = float(s["lon"]), float(s["lat"])
            rgb = None
            for _, ds in datasets:
                rgb = _read_rgb(ds, lon, lat, patch)
                if rgb is not None:
                    break
            if rgb is None:
                continue
            img = Image.fromarray(rgb, mode="RGB")
            img.save(out_dir / f"{s['id']}_{key}.png")
            ok += 1
        for _, ds in datasets:
            ds.close()
        status[key] = {"items": len(datasets), "ok": ok, "samples": len(samples)}
        print(f"    -> {ok}/{len(samples)} patch")

    (base / "labeling").mkdir(parents=True, exist_ok=True)
    (base / "labeling" / "optical_status.json").write_text(
        json.dumps(status, indent=2))
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--sets", nargs="*", default=None, choices=list(SETS))
    ap.add_argument("--patch", type=int, default=PATCH_SIZE)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for city in (a.cities or list(CITIES)):
        print(f"=== {city} ===")
        st = fetch_city(city, a.sets, a.patch, a.limit)
        print(st)


if __name__ == "__main__":
    main()
