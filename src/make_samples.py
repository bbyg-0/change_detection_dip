"""Orkestrator Fase 4: sampling -> patch SAR -> optik -> labeling.html."""

import argparse
import json

from config import CITIES, PATCH_SIZE
from labeling import label_city
from optical import fetch_city
from optical_context import run_city as build_context
from patches import render_city
from sampling import sample_city
from preprocess import output_tile_dir, city_slug


def run_city(city, seed=42, patch=PATCH_SIZE, skip_optical=False,
             optical_sets=None, optical_limit=None):
    print(f"=== {city} ===")
    csv_path, pts = sample_city(city, seed=seed)
    print(f"  sampling: {len(pts)} titik -> {csv_path}")
    n_patch, _ = render_city(city, patch=patch)
    print(f"  patch SAR: {n_patch} sampel")
    if not skip_optical:
        st = fetch_city(city, sets=optical_sets, patch=patch, limit=optical_limit)
        print(f"  optik: {st}")
        ctx = build_context(city, target_km=0.77, patch=patch)
        print(f"  konteks NAIP ~770 m: {ctx}")
    html, n = label_city(city)
    print(f"  labeling: {n} sampel -> {html}")
    return {"city": city, "slug": city_slug(city), "samples": n, "patch": n_patch,
            "html": str(html)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patch", type=int, default=PATCH_SIZE)
    ap.add_argument("--skip-optical", action="store_true")
    ap.add_argument("--optical-sets", nargs="*", default=None)
    ap.add_argument("--optical-limit", type=int, default=None)
    a = ap.parse_args()

    results = []
    for city in (a.cities or list(CITIES)):
        results.append(run_city(city, a.seed, a.patch, a.skip_optical,
                                a.optical_sets, a.optical_limit))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
