"""Estimasi dan penerapan pergeseran (registrasi) antara dua citra.

Konvensi: nilai (dr, dc) yang dikembalikan adalah translasi yang diterapkan
pada citra kedua (b) agar sejajar dengan citra pertama (a), memakai
`scipy.ndimage.shift(b, (dr, dc))`.
"""

import numpy as np
from scipy import ndimage as ndi


def _masked_pearson(x, y, m):
    x = x[m]
    y = y[m]
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if den <= 0:
        return -1.0
    return float(np.sum(x * y) / den)


def _shifted_slices(shape, dr, dc):
    h, w = shape
    ar0, ar1 = max(0, dr), h + min(0, dr)
    br0, br1 = max(0, -dr), h + min(0, -dr)
    ac0, ac1 = max(0, dc), w + min(0, dc)
    bc0, bc1 = max(0, -dc), w + min(0, -dc)
    return (ar0, ar1, ac0, ac1), (br0, br1, bc0, bc1)


def _subpix(rm, r0, rp):
    den = rm - 2.0 * r0 + rp
    return 0.0 if abs(den) < 1e-12 else 0.5 * (rm - rp) / den


def ncc_shift(a, b, max_shift=8, valid=None, min_valid=1000):
    """Pergeseran yang memaksimalkan korelasi silang pada piksel valid."""
    la = np.log10(np.maximum(a, 1e-6)).astype(np.float64)
    lb = np.log10(np.maximum(b, 1e-6)).astype(np.float64)
    if valid is None:
        valid = np.ones(a.shape, dtype=bool)

    table = {}
    best = (-2.0, 0, 0)
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            sa, sb = _shifted_slices(la.shape, dr, dc)
            x = la[sa[0]:sa[1], sa[2]:sa[3]]
            y = lb[sb[0]:sb[1], sb[2]:sb[3]]
            m = valid[sa[0]:sa[1], sa[2]:sa[3]] & valid[sb[0]:sb[1], sb[2]:sb[3]]
            if m.sum() < min_valid:
                continue
            r = _masked_pearson(x, y, m)
            table[(dr, dc)] = r
            if r > best[0]:
                best = (r, dr, dc)

    if best[0] <= -1.0:
        return {"peak_r": float("nan"), "dr": float("nan"), "dc": float("nan"),
                "dr_int": 0, "dc_int": 0, "n_valid": int(valid.sum())}

    _, dr, dc = best
    off_r = off_c = 0.0
    if (dr - 1, dc) in table and (dr + 1, dc) in table:
        off_r = _subpix(table[(dr - 1, dc)], table[(dr, dc)], table[(dr + 1, dc)])
    if (dr, dc - 1) in table and (dr, dc + 1) in table:
        off_c = _subpix(table[(dr, dc - 1)], table[(dr, dc)], table[(dr, dc + 1)])
    # Peak di (dr,dc) adalah translasi yang diterapkan pada b agar sejajar a,
    # yaitu b[y+dr, x+dc] ~ a[y, x]. Terapkan dengan apply_shift_field.
    return {
        "peak_r": best[0],
        "dr": float(dr + off_r),
        "dc": float(dc + off_c),
        "dr_int": int(dr),
        "dc_int": int(dc),
        "n_valid": int(valid.sum()),
    }


def phase_corr_shift(a, b, valid=None):
    """Phase correlation FFT (integer + subpiksel parabolik)."""
    la = np.log10(np.maximum(a, 1e-6)).astype(np.float64)
    lb = np.log10(np.maximum(b, 1e-6)).astype(np.float64)
    if valid is not None and valid.any():
        la = np.where(valid, la, np.median(la[valid]))
        lb = np.where(valid, lb, np.median(lb[valid]))
    win = np.outer(np.hanning(la.shape[0]), np.hanning(la.shape[1]))
    A = np.fft.fft2((la - la.mean()) * win)
    B = np.fft.fft2((lb - lb.mean()) * win)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    c = np.fft.fftshift(np.fft.ifft2(R).real)
    idx = np.unravel_index(np.argmax(c), c.shape)
    dr = idx[0] - la.shape[0] // 2
    dc = idx[1] - la.shape[1] // 2
    off_r = _subpix(c[(idx[0] - 1) % c.shape[0], idx[1]],
                    c[idx], c[(idx[0] + 1) % c.shape[0], idx[1]])
    off_c = _subpix(c[idx[0], (idx[1] - 1) % c.shape[1]],
                    c[idx], c[idx[0], (idx[1] + 1) % c.shape[1]])
    # Peak di (dr,dc): b[y+dr, x+dc] ~ a[y, x], sama konvensi dengan ncc_shift.
    return {"dr": float(dr + off_r), "dc": float(dc + off_c),
            "dr_int": int(dr), "dc_int": int(dc)}


def blockwise_shifts(a, b, valid, block=128, max_shift=4, min_valid=800):
    """Estimasi pergeseran per blok; mengembalikan grid kasar (dr, dc, r, c, ok).

    Blok yang gagal (valid terlalu sedikit / korelasi tidak masuk akal) ditandai
    dengan ok=False sehingga pemanggil bisa mengisinya dengan fallback.
    """
    h, w = a.shape
    ny, nx = -(-h // block), -(-w // block)
    dr_g = np.full((ny, nx), np.nan)
    dc_g = np.full((ny, nx), np.nan)
    ok_g = np.zeros((ny, nx), dtype=bool)
    rr = np.full((ny, nx), np.nan)
    cc = np.full((ny, nx), np.nan)

    for i in range(ny):
        for j in range(nx):
            rs = slice(i * block, min((i + 1) * block, h))
            cs = slice(j * block, min((j + 1) * block, w))
            sub_v = valid[rs, cs]
            if sub_v.sum() < min_valid:
                continue
            res = ncc_shift(a[rs, cs], b[rs, cs], max_shift=max_shift,
                            valid=sub_v, min_valid=min_valid)
            if not np.isfinite(res["dr"]) or res["peak_r"] <= 0.2:
                continue
            dr_g[i, j] = res["dr"]
            dc_g[i, j] = res["dc"]
            ok_g[i, j] = True
            rr[i, j] = (rs.start + rs.stop - 1) / 2.0
            cc[i, j] = (cs.start + cs.stop - 1) / 2.0
    return {"dr": dr_g, "dc": dc_g, "ok": ok_g, "row": rr, "col": cc, "block": block}


def smooth_shift_field(grid, shape, fallback=(0.0, 0.0), order=1):
    """Interpolasi grid pergeseran blok menjadi medan pergeseran per piksel."""
    h, w = shape
    ok = grid["ok"]
    dr_g = grid["dr"].copy()
    dc_g = grid["dc"].copy()

    if not ok.any():
        return np.full(shape, fallback[0], dtype=np.float64), np.full(shape, fallback[1], dtype=np.float64)

    if not ok.all():
        idx_ok = np.argwhere(ok)
        coords_bad = np.argwhere(~ok)
        nearest = np.round(coords_bad[:, None, :] - idx_ok[None, :, :]).astype(int)
        dist2 = (nearest ** 2).sum(axis=-1)
        pick = idx_ok[np.argmin(dist2, axis=1)]
        dr_g[~ok] = dr_g[pick[:, 0], pick[:, 1]]
        dc_g[~ok] = dc_g[pick[:, 0], pick[:, 1]]

    ny, nx = dr_g.shape
    rows = np.arange(h)[:, None] * np.ones((1, w))
    cols = np.ones((h, 1)) * np.arange(w)[None, :]

    dr_map = ndi.map_coordinates(dr_g, [rows / h * (ny - 1e-12), cols / w * (nx - 1e-12)],
                                 order=order, mode="nearest")
    dc_map = ndi.map_coordinates(dc_g, [rows / h * (ny - 1e-12), cols / w * (nx - 1e-12)],
                                 order=order, mode="nearest")
    return dr_map.astype(np.float64), dc_map.astype(np.float64)


def apply_shift_field(img, dr_map, dc_map, order=3):
    """Warp img agar sejajar dengan referensi: out[y,x] = img[y-dr, x-dc]."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    coords = np.array([yy - dr_map, xx - dc_map])
    return ndi.map_coordinates(img.astype(np.float64), coords, order=order, mode="nearest")
