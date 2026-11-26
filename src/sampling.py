"""Sampling titik ground truth: grid 4x4 + oversampling top-5% DI.

Output: output/L1/<slug>/samples/samples.csv dan samples.geojson.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from config import CITIES, OUTPUT_DIR, TILE_SIZE
from preprocess import city_slug, output_tile_dir


def load_di(city):
    slug = city_slug(city)
    base = output_tile_dir(city)
    z = np.load(base / "di" / f"{slug}_di.npz")
    meta = json.loads((base / "prep" / "meta.json").read_text())
    return z["di_norm"].astype(np.float64), z["mask"].astype(bool), meta


def _greedy(cands, n, min_dist, accepted, rng):
    """Pilih n kandidat dengan jarak Euclidean minimum terhadap accepted."""
    if len(cands) == 0 or n <= 0:
        return []
    cands = cands[rng.permutation(len(cands))]
    added = []
    md2 = min_dist * min_dist
    for r, c in cands:
        if len(added) >= n:
            break
        r, c = int(r), int(c)
        ok = True
        for ar, ac in accepted:
            if (r - ar) ** 2 + (c - ac) ** 2 < md2:
                ok = False
                break
        if ok:
            for ar, ac in added:
                if (r - ar) ** 2 + (c - ac) ** 2 < md2:
                    ok = False
                    break
        if ok:
            added.append((r, c))
    return added


def sample_points(di_norm, mask, size=TILE_SIZE, cells=4, per_cell=8,
                  top_frac=0.05, n_top=32, min_dist=10, seed=42):
    """Kembalikan list dict {row, col, di, stratum}."""
    rng = np.random.default_rng(seed)
    valid = mask & np.isfinite(di_norm)
    if not valid.any():
        return []

    cell = size // cells
    accepted, points = [], []
    for gi in range(cells):
        for gj in range(cells):
            r0, c0 = gi * cell, gj * cell
            sub = valid[r0:r0 + cell, c0:c0 + cell]
            coords = np.argwhere(sub) + np.array([r0, c0])
            picked = _greedy(coords, per_cell, min_dist, accepted, rng)
            for r, c in picked:
                accepted.append((r, c))
                points.append({"row": r, "col": c, "di": float(di_norm[r, c]),
                               "stratum": f"grid_{gi}{gj}"})

    # oversampling area DI tertinggi
    vals = di_norm[valid]
    thr = float(np.quantile(vals, 1.0 - top_frac))
    top = np.argwhere(valid & (di_norm >= thr))
    picked = _greedy(top, n_top, min_dist, accepted, rng)
    for r, c in picked:
        accepted.append((r, c))
        points.append({"row": int(r), "col": int(c), "di": float(di_norm[r, c]),
                       "stratum": "top5"})

    target = cells * cells * per_cell + n_top
    if len(points) < target:
        allc = np.argwhere(valid)
        for md in (min_dist, max(4, min_dist // 2), 1):
            need = target - len(points)
            if need <= 0:
                break
            extra = _greedy(allc, need, md, accepted, rng)
            for r, c in extra:
                accepted.append((r, c))
                points.append({"row": int(r), "col": int(c),
                               "di": float(di_norm[r, c]), "stratum": "fill"})
    return points


def add_geo(points, meta, slug):
    """Tambahkan id, lat/lon, dan indeks global."""
    gt = meta["geotransform"]
    tile = meta["tile"]
    for i, p in enumerate(points):
        p["id"] = f"{slug}_{i:04d}"
        p["city"] = meta["city"]
        p["lat"] = gt["north"] + (p["row"] + 0.5) * gt["dy"]
        p["lon"] = gt["west"] + (p["col"] + 0.5) * gt["dx"]
        p["global_row"] = tile["r0"] + p["row"]
        p["global_col"] = tile["c0"] + p["col"]
    return points


def write_samples(points, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["id", "city", "row", "col", "global_row", "global_col",
            "lat", "lon", "di", "stratum"]
    csv_path = out_dir / "samples.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for p in points:
            w.writerow({k: p[k] for k in cols})

    geo = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature",
             "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
             "properties": {k: p[k] for k in cols}}
            for p in points
        ],
    }
    (out_dir / "samples.geojson").write_text(json.dumps(geo), encoding="utf-8")
    return csv_path


def sample_city(city, seed=42, **kw):
    di_norm, mask, meta = load_di(city)
    h, w = di_norm.shape
    pts = sample_points(di_norm, mask, size=min(h, w), **kw)
    pts = add_geo(pts, meta, city_slug(city))
    out_dir = output_tile_dir(city) / "samples"
    csv_path = write_samples(pts, out_dir)
    return csv_path, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--top-frac", type=float, default=0.05)
    ap.add_argument("--n-top", type=int, default=32)
    ap.add_argument("--min-dist", type=float, default=10.0)
    a = ap.parse_args()
    cities = a.cities or list(CITIES)
    for city in cities:
        p, pts = sample_city(city, seed=a.seed, per_cell=a.per_cell,
                             top_frac=a.top_frac, n_top=a.n_top,
                             min_dist=a.min_dist)
        strata = {}
        for x in pts:
            strata[x["stratum"]] = strata.get(x["stratum"], 0) + 1
        print(f"{city}: {len(pts)} titik -> {p}  strata={strata}")


if __name__ == "__main__":
    main()
