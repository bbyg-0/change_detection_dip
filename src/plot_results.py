"""Figur hasil change detection: DI, peta perubahan, histogram + kurva J(T)."""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import OUTPUT_DIR


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def plot_di(base, slug):
    z = np.load(base / "di" / f"{slug}_di.npz")
    di = np.where(z["mask"].astype(bool), z["di"], np.nan)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(np.log10(np.maximum(di, 1e-3)), cmap="inferno")
    ax.set_title(f"DI omnibus (log10) - {slug}")
    fig.colorbar(im, ax=ax, label="log10(DI)")
    out = base / "figures" / f"{slug}_di.png"
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_change(base, slug):
    outs = []
    cmap_dir = base / "change_map"
    for npz in sorted(cmap_dir.glob("*.npz")):
        z = np.load(npz)
        if "change" not in z.files:
            continue
        arr = z["change"].astype(np.uint8)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
        ax.set_title(npz.stem.replace(f"{slug}_", ""))
        ax.axis("off")
        out = base / "figures" / f"{npz.stem}_map.png"
        fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
        outs.append(out)
    return outs


def plot_ki(base, slug, pdf):
    npz = base / "change_map" / f"{slug}_{pdf}_ki.npz"
    if not npz.exists():
        return None
    z = np.load(npz)
    T = int(z["T"])
    Js = z["Js"].astype(np.float64)
    hist = z["hist"].astype(np.float64)
    L = len(Js)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist, lw=0.8)
    axes[0].axvline(T, color="r", ls="--", label=f"T*={T}")
    axes[0].set_title(f"Histogram DI ({pdf})")
    axes[0].set_xlabel("level"); axes[0].legend()
    axes[1].plot(Js, lw=0.8)
    axes[1].axvline(T, color="r", ls="--")
    axes[1].set_title(f"J(T) ({pdf})")
    axes[1].set_xlabel("T")
    fig.tight_layout()
    out = base / "figures" / f"{slug}_{pdf}_ki.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="*", default=None)
    a = ap.parse_args()
    if a.cities:
        slugs = [slugify(c) for c in a.cities]
    else:
        slugs = sorted(p.name for p in (OUTPUT_DIR / "L1").iterdir() if p.is_dir())
    for slug in slugs:
        base = OUTPUT_DIR / "L1" / slug
        (base / "figures").mkdir(parents=True, exist_ok=True)
        made = [plot_di(base, slug)]
        made += plot_change(base, slug)
        for pdf in ("gamma", "weibull", "gaussian"):
            p = plot_ki(base, slug, pdf)
            if p:
                made.append(p)
        print(f"{slug}: {len(made)} figur")


if __name__ == "__main__":
    main()
