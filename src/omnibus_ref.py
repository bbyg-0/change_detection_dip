"""Referensi Python untuk DI omnibus (Eq. 10) — dipakai sebagai parity test.

Bukan bagian pipeline utama; pipeline utama ada di Julia.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from config import LOOKS, N_DATES, POL_DIM, SQRT2


def load_prep(npz_path):
    z = np.load(npz_path)
    return {k: z[k] for k in z.files}


def build_covariance(z, ds):
    hh = z[f"{ds}_HHHH"].astype(np.float64)
    hv = z[f"{ds}_HVHV"].astype(np.float64)
    vv = z[f"{ds}_VVVV"].astype(np.float64)
    c12 = SQRT2 * z[f"{ds}_HHHV"].astype(np.complex128)
    c13 = z[f"{ds}_HHVV"].astype(np.complex128)
    c23 = SQRT2 * z[f"{ds}_HVVV"].astype(np.complex128)

    C = np.empty(hh.shape + (3, 3), dtype=np.complex128)
    C[..., 0, 0] = hh
    C[..., 1, 1] = 2.0 * hv
    C[..., 2, 2] = vv
    C[..., 0, 1] = c12
    C[..., 0, 2] = c13
    C[..., 1, 2] = c23
    C[..., 1, 0] = np.conj(c12)
    C[..., 2, 0] = np.conj(c13)
    C[..., 2, 1] = np.conj(c23)
    return C


def diagonal_loading(C, eps=1e-6):
    out = C.copy()
    tr = np.real(np.trace(C, axis1=-2, axis2=-1)) / C.shape[-1]
    idx = np.arange(C.shape[-1])
    out[..., idx, idx] += (eps * tr)[..., None]
    return out


def det_real(C):
    d = np.linalg.det(C)
    return np.maximum(d.real, 1e-30)


def omnibus_raw(C1, C2, p=POL_DIM, m=LOOKS, n=N_DATES):
    """Nilai mentah Eq. 10 (belum diorientasikan)."""
    coeff = (2.0 * p * p - 4.0 * p * m - 1.0) / (2.0 * p)
    logdet1 = np.log(det_real(C1))
    logdet2 = np.log(det_real(C2))
    logdet_sum = np.log(det_real(C1 + C2))
    bracket = p * n * np.log(n) + logdet1 + logdet2 - n * logdet_sum
    return coeff * bracket


def psd_ok(C, tol=1e-8):
    """Mask PSD per piksel (eigenvalue terkecil >= -tol*trace/p)."""
    e = np.linalg.eigvalsh(C)
    s = np.maximum(np.real(np.trace(C, axis1=-2, axis2=-1)) / C.shape[-1], 1e-30)
    return e[..., 0] >= -tol * s


def compute_di(prep_npz, eps=1e-6, orientation="neg", psd_mask=True):
    z = load_prep(prep_npz)
    mask = z["mask"].astype(bool)
    C1 = diagonal_loading(build_covariance(z, "T1"), eps)
    C2 = diagonal_loading(build_covariance(z, "T5"), eps)
    if psd_mask:
        ok1 = psd_ok(C1)
        ok2 = psd_ok(C2)
        new_mask = mask.copy()
        idx = np.flatnonzero(mask)
        new_mask.flat[idx[~(ok1[mask] & ok2[mask])]] = False
        mask = new_mask
    raw = omnibus_raw(C1, C2)
    di = -raw if orientation == "neg" else raw
    di = np.where(mask, di, np.nan)

    lo = float(np.nanmin(di))
    hi = float(np.nanmax(di))
    norm = (di - lo) / (hi - lo) if hi > lo else np.zeros_like(di)
    return {
        "raw": raw,
        "di": di,
        "di_norm": norm,
        "mask": mask,
        "min": lo,
        "max": hi,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prep_npz")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--orientation", choices=["neg", "pos"], default="neg")
    ap.add_argument("--save", default=None, help="output .npz")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()

    r = compute_di(a.prep_npz, a.eps, a.orientation)
    if a.stats:
        di = r["di"][r["mask"]]
        qs = [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
        print("valid pixels:", di.size, f"({di.size / r['mask'].size * 100:.2f}%)")
        print("min/max:", r["min"], r["max"])
        for q in qs:
            print(f"  p{q:>5}: {np.percentile(di, q):.6g}")
        # mode kasar
        hist, edges = np.histogram(di, bins=50)
        k = int(np.argmax(hist))
        print(f"  mode hist: {0.5*(edges[k]+edges[k+1]):.6g}")
    if a.save:
        np.savez_compressed(a.save, di_raw=r["raw"].astype(np.float32),
                            di=r["di"].astype(np.float32),
                            di_norm=r["di_norm"].astype(np.float32),
                            mask=r["mask"].astype(np.uint8),
                            min=np.float64(r["min"]), max=np.float64(r["max"]))
        print("saved", a.save)


if __name__ == "__main__":
    main()
