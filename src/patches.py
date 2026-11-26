"""Render patch SAR 128x128 (RGB palsu HH/HV/VV) + crosshair."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import CITIES, PATCH_SIZE
from preprocess import output_tile_dir

CHANNELS = ("HHHH", "HVHV", "VVVV")


def load_prep(city):
    base = output_tile_dir(city)
    z = np.load(base / "prep" / "prep.npz")
    meta = json.loads((base / "prep" / "meta.json").read_text())
    return {k: z[k] for k in z.files}, meta


def compute_stretch(z, mask):
    """Stretch dB per kanal dari kedua tanggal (persentil 2-98)."""
    stretch = {}
    for band in CHANNELS:
        vals = []
        for date in ("T1", "T5"):
            x = z[f"{date}_{band}"].astype(np.float64)[mask]
            x = x[np.isfinite(x) & (x > 0)]
            vals.append(10.0 * np.log10(np.maximum(x, 1e-6)))
        db = np.concatenate(vals)
        lo, hi = np.percentile(db, [2, 98])
        stretch[band] = (float(lo), float(hi) if hi > lo else float(lo + 1.0))
    return stretch


def crosshair_mask(size, cy, cx):
    """Mask piksel yang pernah digambar crosshair (untuk dibersihkan)."""
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.line([(cx - 10, cy), (cx - 3, cy)], fill=255, width=1)
    d.line([(cx + 3, cy), (cx + 10, cy)], fill=255, width=1)
    d.line([(cx, cy - 10), (cx, cy - 3)], fill=255, width=1)
    d.line([(cx, cy + 3), (cx, cy + 10)], fill=255, width=1)
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=255, width=1)
    return np.array(m) > 0


def remove_crosshair(img, cy, cx, color=(255, 0, 0), opacity=0.5):
    """Balikkan alpha-blend crosshair lama sehingga patch kembali bersih."""
    arr = np.array(img).astype(np.float64)
    m = crosshair_mask(img.size, cy, cx)
    a = round(255 * opacity) / 255.0
    src = np.array(color, dtype=np.float64)
    arr[m] = (arr[m] - a * src) / (1.0 - a)
    arr = np.clip(arr, 0, 255)
    return Image.fromarray(arr.round().astype(np.uint8))


def render_sar(z, date, row, col, stretch, patch=PATCH_SIZE):
    h, w = z[f"{date}_HHHH"].shape
    top = int(np.clip(row - patch // 2, 0, max(0, h - patch)))
    left = int(np.clip(col - patch // 2, 0, max(0, w - patch)))
    chans = []
    for band in CHANNELS:
        x = z[f"{date}_{band}"][top:top + patch, left:left + patch].astype(np.float64)
        db = 10.0 * np.log10(np.maximum(x, 1e-6))
        lo, hi = stretch[band]
        chans.append(np.clip((db - lo) / (hi - lo), 0.0, 1.0))
    rgb = (np.stack(chans, axis=-1) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def render_city(city, patch=PATCH_SIZE):
    base = output_tile_dir(city)
    meta = json.loads((base / "prep" / "meta.json").read_text())
    z, _ = load_prep(city)
    mask = z["T1_HHHH"] > 0
    for b in CHANNELS:
        mask &= z[f"T5_{b}"] > 0
    stretch = compute_stretch(z, mask)

    with open(base / "samples" / "samples.csv", newline="") as f:
        samples = list(csv.DictReader(f))

    out_dir = base / "patches"
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in samples:
        row, col = int(s["row"]), int(s["col"])
        for date in ("T1", "T5"):
            img = render_sar(z, date, row, col, stretch, patch=patch)
            img.save(out_dir / f"{s['id']}_{date}.png")
    return len(samples), out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--patch", type=int, default=PATCH_SIZE)
    a = ap.parse_args()
    for city in (a.cities or list(CITIES)):
        n, out = render_city(city, patch=a.patch)
        print(f"{city}: {n} sample -> {out}")


if __name__ == "__main__":
    main()
