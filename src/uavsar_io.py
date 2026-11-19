"""Pembacaan file UAVSAR (annotation dan raw GRD) secara windowed."""

import re

import numpy as np

from config import CROSS_BANDS, DATASET_KEYS, DATASETS, POWER_BANDS


def read_ann(ann_path):
    """Parse baris 'keyword (units) = value ; comment' menjadi dict."""
    params = {}
    with open(ann_path, errors="ignore") as f:
        for line in f:
            line = line.split(";")[0]
            m = re.match(r"\s*(.*?)\s*(?:\([^)]*\))?\s*=\s*(.+)", line)
            if m and m.group(1).strip():
                params[m.group(1).strip()] = m.group(2).strip()
    return params


def ann_path(dataset):
    d = DATASETS[dataset]
    return d["dir"] / f"{d['prefix']}_{d['cx']}.ann"


def ann_info(dataset):
    """Kembalikan rows, cols, dx, dy, west, north, looks."""
    a = read_ann(ann_path(dataset))
    rows = int(float(a["grd_mag.set_rows"]))
    cols = int(float(a["grd_mag.set_cols"]))
    dx = abs(float(a["grd_mag.col_mult"]))
    dy = abs(float(a["grd_mag.row_mult"]))
    west = float(a["grd_mag.col_addr"]) - dx / 2
    north = float(a["grd_mag.row_addr"]) + dy / 2
    looks = None
    try:
        looks = int(float(a["Number of Range Looks in GRD"])) * int(
            float(a["Number of Azimuth Looks in GRD"])
        )
    except KeyError:
        pass
    return {
        "rows": rows,
        "cols": cols,
        "dx": dx,
        "dy": dy,
        "west": west,
        "north": north,
        "looks": looks,
        "params": a,
    }


def band_path(dataset, key):
    d = DATASETS[dataset]
    if key in POWER_BANDS or key in CROSS_BANDS:
        return d["dir"] / f"{d['prefix']}{key}_{d['cx']}.grd"
    if key in ("inc", "slope", "ann"):
        return d["dir"] / f"{d['prefix']}_{d['cx']}.{key}"
    raise KeyError(f"Band tidak dikenal: {key}")


def band_dtype(key, info):
    if key in POWER_BANDS:
        return "<f4"
    if key in CROSS_BANDS:
        return "<c8"
    if key == "inc":
        return "<f4"
    if key == "slope":
        return "<f4"
    raise KeyError(key)


def read_tile(dataset, key, r0, c0, size, info=None):
    """Baca potongan size x size dari file raw menggunakan memmap (windowed)."""
    if info is None:
        info = ann_info(dataset)
    rows, cols = info["rows"], info["cols"]
    path = band_path(dataset, key)
    dtype = band_dtype(key, info)

    if key == "slope":
        mm = np.memmap(path, dtype=dtype, mode="r", shape=(rows, cols, 2))
        tile = np.array(mm[r0:r0 + size, c0:c0 + size, :])
    else:
        mm = np.memmap(path, dtype=dtype, mode="r", shape=(rows, cols))
        tile = np.array(mm[r0:r0 + size, c0:c0 + size])
    del mm
    return tile


def read_tile_pair(name, city, size=None):
    """Baca semua band yang diperlukan untuk satu tile pada grid overlap."""
    from config import city_tile

    if city is None:
        city = city_tile(name, size) if size else city_tile(name)
    size = city["size"]
    out = {}
    for ds in DATASET_KEYS:
        info = ann_info(ds)
        r0, c0 = city["crops"][ds]
        bands = {}
        for key in POWER_BANDS + CROSS_BANDS + ("inc", "slope"):
            bands[key] = read_tile(ds, key, r0, c0, size, info)
        out[ds] = bands
    return out, city
