"""Pembentukan matriks kovarians PolSAR dari produk GRD UAVSAR."""

import numpy as np

from config import SQRT2


def build_covariance(bands, factors=True):
    """Susun matriks kovarians 3x3 Hermitian per piksel.

    Omega = [HH, sqrt(2) HV, VV]^T sehingga:
        C11 = |HH|^2, C22 = 2|HV|^2, C33 = |VV|^2
        C12 = sqrt(2) <HH HV*>, C13 = <HH VV*>, C23 = sqrt(2) <HV VV*>

    Jika factors=False, dipakai tanpa faktor sqrt(2)/2 (untuk pembanding).
    """
    hhhh = bands["HHHH"].astype(np.float64)
    hvhv = bands["HVHV"].astype(np.float64)
    vvvv = bands["VVVV"].astype(np.float64)
    hhhv = bands["HHHV"].astype(np.complex128)
    hhvv = bands["HHVV"].astype(np.complex128)
    hvvv = bands["HVVV"].astype(np.complex128)

    if factors:
        c22 = 2.0 * hvhv
        c12 = SQRT2 * hhhv
        c23 = SQRT2 * hvvv
    else:
        c22 = hvhv
        c12 = hhhv
        c23 = hvvv

    shape = hhhh.shape
    C = np.empty(shape + (3, 3), dtype=np.complex128)
    C[..., 0, 0] = hhhh
    C[..., 1, 1] = c22
    C[..., 2, 2] = vvvv
    C[..., 0, 1] = c12
    C[..., 0, 2] = hhvv
    C[..., 1, 2] = c23
    C[..., 1, 0] = np.conj(c12)
    C[..., 2, 0] = np.conj(hhvv)
    C[..., 2, 1] = np.conj(c23)
    return C


def valid_mask(bands):
    """Piksel valid: semua diagonal finite dan > 0, cross-term finite."""
    diag = [bands[k].astype(np.float64) for k in ("HHHH", "HVHV", "VVVV")]
    ok = np.ones(diag[0].shape, dtype=bool)
    for d in diag:
        ok &= np.isfinite(d) & (d > 0)
    for k in ("HHHV", "HHVV", "HVVV"):
        ok &= np.isfinite(bands[k].real) & np.isfinite(bands[k].imag)
    return ok


def correlation_stats(C, mask):
    """Statistik korelasi ternormalisasi |Cij|/sqrt(Cii*Cjj)."""
    out = {}
    for (i, j, name) in ((0, 1, "r12"), (0, 2, "r13"), (1, 2, "r23")):
        num = np.abs(C[..., i, j])[mask]
        den = np.sqrt(C[..., i, i].real[mask] * C[..., j, j].real[mask])
        r = num / np.maximum(den, 1e-30)
        out[name] = {
            "mean": float(np.mean(r)),
            "median": float(np.median(r)),
            "p99": float(np.percentile(r, 99)),
            "max": float(np.max(r)),
            "frac_gt_1": float(np.mean(r > 1.0 + 1e-6)),
        }
    return out


def psd_stats(C, mask, tol=1e-6):
    """Fraksi piksel dengan matriks kovarians positif semidefinite."""
    sub = C[mask]
    eig = np.linalg.eigvalsh(sub)
    scale = np.maximum(np.abs(eig[:, -1]), 1e-30)
    min_eig = eig[:, 0]
    return {
        "n": int(sub.shape[0]),
        "frac_psd": float(np.mean(min_eig >= -tol * scale)),
        "min_eig_median": float(np.median(min_eig)),
        "min_eig_min": float(np.min(min_eig)),
    }


def determinant_imag_stats(C, mask):
    """Det(C) harus real untuk matriks Hermitian; ukur sisa bagian imajiner."""
    sub = C[mask]
    det = np.linalg.det(sub)
    scale = np.maximum(np.abs(det), 1e-30)
    rel_imag = np.abs(det.imag) / scale
    return {
        "max_abs_imag": float(np.max(np.abs(det.imag))),
        "max_rel_imag": float(np.max(rel_imag)),
        "median_rel_imag": float(np.median(rel_imag)),
    }
