"""Patch konteks NAIP ~770 m dari aset rendered_preview.

Preview tidak ter-georeferensi, tetapi item menyediakan proj:transform,
proj:shape, dan proj:code. Pemetaan lon/lat -> piksel preview dilakukan
melalui CRS asli (bukan linear lon/lat) agar posisi crosshair akurat.
"""

import argparse
import csv
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import planetary_computer as pc
import requests
from PIL import Image
from pystac_client import Client
from rasterio.warp import transform as warp_transform

from config import CITIES, PATCH_SIZE
from optical import SETS, _search
from preprocess import output_tile_dir

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEFAULT_KM = 0.77


def _load_preview(item):
    signed = pc.sign(item)
    r = requests.get(signed.assets["rendered_preview"].href, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def _native_geometry(item):
    """Kembalikan (code, x0, y0, x1, y1, pw, ph) untuk preview item."""
    tr = item.properties["proj:transform"]
    shape = item.properties["proj:shape"]
    code = item.properties["proj:code"]
    a, _, c, _, e, f = tr[:6]
    nw, nh = shape[1], shape[0]
    x0, y0 = c, f
    x1, y1 = c + a * nw, f + e * nh
    return code, x0, y0, x1, y1


def _project_point(item, lon, lat, img):
    """Posisi titik pada piksel preview + jarak ternormalisasi ke pusat."""
    code, x0, y0, x1, y1 = _native_geometry(item)
    x, y = warp_transform("EPSG:4326", code, [lon], [lat])
    x, y = x[0], y[0]
    xmin, xmax = min(x0, x1), max(x0, x1)
    ymin, ymax = min(y0, y1), max(y0, y1)
    if not (xmin <= x <= xmax and ymin <= y <= ymax):
        return None
    pw, ph = img.size
    px = (x - x0) / (x1 - x0) * pw
    py = (y0 - y) / (y0 - y1) * ph
    cx, cy = pw / 2.0, ph / 2.0
    d = ((px - cx) / pw) ** 2 + ((py - cy) / ph) ** 2
    return {"px": px, "py": py, "pw": pw, "ph": ph,
            "gpx": abs((x1 - x0) / pw), "gpy": abs((y1 - y0) / ph), "d": d}


def _crop(img, proj, target_km, patch):
    cw = target_km * 1000.0 / max(proj["gpx"], 1e-9)
    ch = target_km * 1000.0 / max(proj["gpy"], 1e-9)
    cw = min(max(cw, 2.0), proj["pw"])
    ch = min(max(ch, 2.0), proj["ph"])
    left = min(max(proj["px"] - cw / 2.0, 0.0), max(0.0, proj["pw"] - cw))
    top = min(max(proj["py"] - ch / 2.0, 0.0), max(0.0, proj["ph"] - ch))
    crop = img.crop((round(left), round(top), round(left + cw), round(top + ch)))
    crop = crop.resize((patch, patch), Image.LANCZOS)
    cx = (proj["px"] - left) / cw * 100.0
    cy = (proj["py"] - top) / ch * 100.0
    return crop, float(cx), float(cy)


def _city_bbox(base):
    meta = json.loads((base / "prep" / "meta.json").read_text())
    gt = meta["geotransform"]
    h, w = meta["tile"]["h"], meta["tile"]["w"]
    west, north = gt["west"], gt["north"]
    east = west + w * gt["dx"]
    south = north + h * gt["dy"]
    return [min(west, east), min(north, south), max(west, east), max(north, south)]


def run_city(city, target_km=DEFAULT_KM, patch=PATCH_SIZE,
             sets=("NAIP_2020", "NAIP_2022")):
    base = output_tile_dir(city)
    with open(base / "samples" / "samples.csv") as f:
        samples = list(csv.DictReader(f))
    catalog = Client.open(CATALOG_URL)
    out_dir = base / "patches"
    out_dir.mkdir(parents=True, exist_ok=True)
    positions = {s["id"]: {} for s in samples}

    for key in sets:
        cfg = SETS[key]
        items = _search(catalog, cfg["collection"], _city_bbox(base),
                        cfg["datetime"], cloud=None, limit=10)
        previews = []
        for it in items:
            try:
                previews.append((it, _load_preview(it)))
            except Exception as e:
                print(f"  ! preview gagal {it.id}: {str(e)[:80]}")
        print(f"  {city} / {key}: {len(previews)} preview")
        ok = 0
        for s in samples:
            lon, lat = float(s["lon"]), float(s["lat"])
            best = None
            for it, img in previews:
                proj = _project_point(it, lon, lat, img)
                if proj is None:
                    continue
                if best is None or proj["d"] < best[0]["d"]:
                    best = (proj, img)
            if best is None:
                continue
            crop, cx, cy = _crop(best[1], best[0], target_km, patch)
            crop.save(out_dir / f"{s['id']}_{key}_ctx.png")
            positions[s["id"]][key] = {"cx_pct": cx, "cy_pct": cy}
            ok += 1
        print(f"    -> {ok}/{len(samples)} konteks")

    out = base / "labeling" / "optical_context.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"target_km": target_km, "positions": positions}, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--km", type=float, default=DEFAULT_KM)
    ap.add_argument("--patch", type=int, default=PATCH_SIZE)
    a = ap.parse_args()
    for city in (a.cities or list(CITIES)):
        print(f"=== {city} ===")
        print(run_city(city, a.km, a.patch))


if __name__ == "__main__":
    main()
