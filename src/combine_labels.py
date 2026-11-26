"""Gabungkan CSV hasil labeling (per kota) menjadi labels/L1_labels.csv."""

import argparse
import csv
from pathlib import Path

from config import LABELS_DIR


FIELDS = ["id", "city", "row", "col", "global_row", "global_col",
          "lat", "lon", "label", "note", "timestamp"]


def load_rows(paths):
    rows = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if not r.get("id"):
                    continue
                old = rows.get(r["id"])
                if old is None or r.get("timestamp", "") >= old.get("timestamp", ""):
                    rows[r["id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="CSV hasil export dari labeling.html")
    ap.add_argument("--out", default=str(LABELS_DIR / "L1_labels.csv"))
    a = ap.parse_args()

    paths = []
    for p in a.inputs:
        pp = Path(p)
        if pp.is_dir():
            paths += sorted(pp.glob("labels_*.csv"))
        else:
            paths.append(pp)
    rows = load_rows(paths)
    ordered = sorted(rows.values(), key=lambda r: (r.get("city", ""),
                                                   int(r.get("global_row", 0)),
                                                   int(r.get("global_col", 0))))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in ordered:
            w.writerow({k: r.get(k, "") for k in FIELDS})

    counts = {"1": 0, "0": 0, "u": 0, "": 0}
    for r in ordered:
        counts[r.get("label", "")] = counts.get(r.get("label", ""), 0) + 1
    print(f"Gabung {len(paths)} file -> {len(ordered)} label -> {out}")
    print(f"  berubah={counts.get('1',0)}, tidak={counts.get('0',0)}, "
          f"ragu={counts.get('u',0)}, kosong={counts.get('',0)}")


if __name__ == "__main__":
    main()
