"""Konfigurasi global eksperimen change detection UAVSAR P-band Eel River."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LOCATION = "L1"
OUTPUT_DIR = ROOT / "output"
LABELS_DIR = ROOT / "labels"
PR_DIR = ROOT / "PR"

# ---------------------------------------------------------------------------
# Dataset UAVSAR (produk GRD 0.5 arcsec, quad-pol).
# Hanya dua tanggal yang dipakai (n=2): T1 (2020) dan T5 (2023).
# Offset baris/kolom adalah posisi UL tiap tanggal relatif terhadap grid T5.
# ---------------------------------------------------------------------------
DATASETS = {
    "T1": {
        "dir": DATA_DIR / LOCATION / "T1",
        "prefix": "eelriv_15807_20037_012_201113_PL09043020_05",
        "cx": "CX_02",
        "date": "2020-11-13",
        "row_offset": 246,
        "col_offset": 414,
    },
    "T5": {
        "dir": DATA_DIR / LOCATION / "T5",
        "prefix": "eelriv_15807_23014_024_230913_PL09043020_05",
        "cx": "CX_01",
        "date": "2023-09-13",
        "row_offset": 0,
        "col_offset": 0,
    },
}
DATASET_KEYS = tuple(DATASETS)

POWER_BANDS = ("HHHH", "HVHV", "VVVV")
CROSS_BANDS = ("HHHV", "HHVV", "HVVV")

# ---------------------------------------------------------------------------
# Grid overlap (grid T5). Offset T1 -> T5 didapat dari .ann:
# offset kolom +414, offset baris +246 (bilangan bulat, tanpa resampling).
# ---------------------------------------------------------------------------
GRID = {
    "north": 41.284166666666664,
    "west": -124.31666666666666,
    "pixel_deg": 0.0001388888888888889,
    "rows": 6031,
    "cols": 4561,
}
T1_ROW_OFFSET = DATASETS["T1"]["row_offset"]
T1_COL_OFFSET = DATASETS["T1"]["col_offset"]

# ---------------------------------------------------------------------------
# Tile per kota (koordinat pusat dari OpenStreetMap/Nominatim).
# Eureka & Samoa diganti Trinidad & Blue Lake karena di luar swath UAVSAR.
# ---------------------------------------------------------------------------
TILE_SIZE = 512
PATCH_SIZE = 128

CITIES = {
    "Tyee City": (40.9223483, -124.1267316),
    "Arcata": (40.8665170, -124.0828400),
    "McKinleyville": (40.9675585, -124.1032897),
    "Trinidad": (41.0592910, -124.1431250),
    "Blue Lake": (40.8829070, -123.9839490),
}

# ---------------------------------------------------------------------------
# Parameter PolSAR
# ---------------------------------------------------------------------------
POL_DIM = 3          # p
LOOKS_RANGE = 7
LOOKS_AZIMUTH = 27
LOOKS = LOOKS_RANGE * LOOKS_AZIMUTH  # m = 189
N_DATES = len(DATASET_KEYS)          # n = 2

SQRT2 = 2.0 ** 0.5


def city_tile(name, size=TILE_SIZE):
    """Kembalikan indeks tile pada grid overlap dan grid asli tiap tanggal.

    Returns dict dengan kunci: name, r0, c0, r1, c1 (grid overlap),
    crops (dict tanggal -> (row, col) pada file asli), lat, lon, size.
    """
    lat, lon = CITIES[name]
    row_center = (GRID["north"] - lat) / GRID["pixel_deg"]
    col_center = (lon - GRID["west"]) / GRID["pixel_deg"]
    half = size // 2
    r0 = int(round(row_center)) - half
    c0 = int(round(col_center)) - half
    r1, c1 = r0 + size, c0 + size
    if r0 < 0 or c0 < 0 or r1 > GRID["rows"] or c1 > GRID["cols"]:
        raise ValueError(f"Tile {name} keluar dari area overlap: {(r0, c0, r1, c1)}")

    crops = {
        ds: (r0 + meta["row_offset"], c0 + meta["col_offset"])
        for ds, meta in DATASETS.items()
    }
    return {
        "name": name,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "crops": crops,
        "lat": lat,
        "lon": lon,
        "size": size,
    }
