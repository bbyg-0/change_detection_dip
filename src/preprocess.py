"""Fase 1: koreksi subpiksel co-registration dan penyiapan data per tile.

Alur per kota/tile:
1. Baca tile T1 dan T5 (windowed, overlap T1/T5).
2. Bangun masker valid gabungan (kedua waktu).
3. Estimasi residual pergeseran T5 terhadap T1 pada kanal HH (dB).
   - global: 1 offset (dr,dc) untuk seluruh tile.
   - blockwise: medan pergeseran per blok, diinterpolasi ke piksel.
4. Terapkan koreksi ke semua kanal T5, re-mask border hasil warp.
5. (Opsional) diagonal loading kecil agar kovarians definit positif.
6. Kembalikan band T1/T5, masker, dan metadata pergeseran.

Konvensi: apply_shift_field(img, dr, dc) menghasilkan img[y-dr, x-dc],
yang memulihkan img agar sejajar dengan referensi.
"""

import json
import re
from pathlib import Path

import numpy as np

from config import (
    CITIES,
    CROSS_BANDS,
    DATASETS,
    GRID,
    LOOKS,
    N_DATES,
    OUTPUT_DIR,
    POL_DIM,
    POWER_BANDS,
    PR_DIR,
    city_tile,
)

ALL_BANDS = POWER_BANDS + CROSS_BANDS
from covariance import build_covariance, determinant_imag_stats, psd_stats, valid_mask
from registration import apply_shift_field, blockwise_shifts, ncc_shift, smooth_shift_field
from uavsar_io import read_tile_pair


def city_slug(name):
    """'Tyee City' -> 'tyee_city'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def output_tile_dir(city):
    return OUTPUT_DIR / "L1" / city_slug(city)


def estimate_shift(tile_t1, tile_t5, valid, mode="global", block=128, max_shift=8):
    """Estimasi pergeseran T5->T1 dari kanal HH (power)."""
    a = tile_t1["HHHH"]
    b = tile_t5["HHHH"]
    global_res = ncc_shift(a, b, max_shift=max_shift, valid=valid)

    if mode == "blockwise":
        grid = blockwise_shifts(a, b, valid, block=block, max_shift=max(4, max_shift // 2))
        n_ok = int(grid["ok"].sum())
        total = grid["ok"].size
        if n_ok < max(1, total // 4):
            mode = "global"
        grid_info = {
            "n_ok": n_ok, "n_total": total, "block": block,
            "dr_ok_mean": float(np.nanmean(grid["dr"][grid["ok"]])) if n_ok else None,
            "dc_ok_mean": float(np.nanmean(grid["dc"][grid["ok"]])) if n_ok else None,
        }
    else:
        grid = None
        grid_info = None

    return {
        "mode": mode,
        "global": global_res,
        "grid": grid,
        "grid_info": grid_info,
    }


def apply_correction(tile_t5, shift_info, valid):
    """Koreksi semua kanal T5; kembalikan tile baru dan masker border."""
    h, w = tile_t5["HHHH"].shape
    if shift_info["mode"] == "blockwise" and shift_info["grid"] is not None:
        dr_map, dc_map = smooth_shift_field(shift_info["grid"], (h, w),
                                            fallback=(shift_info["global"]["dr"],
                                                      shift_info["global"]["dc"]))
    else:
        g = shift_info["global"]
        dr_map = np.full((h, w), g["dr"], dtype=np.float64)
        dc_map = np.full((h, w), g["dc"], dtype=np.float64)

    corrected = {}
    for band in ALL_BANDS:
        arr = tile_t5[band]
        if np.iscomplexobj(arr):
            re = apply_shift_field(arr.real, dr_map, dc_map, order=3)
            im = apply_shift_field(arr.imag, dr_map, dc_map, order=3)
            corrected[band] = re + 1j * im
        else:
            corrected[band] = np.maximum(apply_shift_field(arr, dr_map, dc_map, order=3), 0.0)

    # border: piksel yang keluar dari domain setelah shift dianggap invalid
    yy, xx = np.mgrid[0:h, 0:w]
    src_y = yy - dr_map
    src_x = xx - dc_map
    in_bounds = (src_y >= 0) & (src_y <= h - 1) & (src_x >= 0) & (src_x <= w - 1)

    # validitas pasca-warp: power band hasil harus positif/terbaca
    warped_valid = np.ones((h, w), dtype=bool)
    for b in POWER_BANDS:
        v = corrected[b]
        warped_valid &= np.isfinite(v) & (v > 0)
    return corrected, in_bounds & warped_valid, dr_map, dc_map


def diagonal_loading(covs, eps=1e-6):
    """Tambah eps * trace(C)/p pada diagonal per piksel (stabilisasi numerik)."""
    if eps <= 0:
        return covs
    out = covs.copy()
    for k, C in enumerate(out):
        tr = np.real(np.trace(C, axis1=-2, axis2=-1)) / C.shape[-1]
        load = (eps * tr)[..., None]
        idx = np.arange(C.shape[-1])
        C[..., idx, idx] += load
        out[k] = C
    return out


def estimate_registration_city(city_name, mode="auto", block=128, max_shift=8):
    """Estimasi pergeseran untuk satu kota; tidak mengubah data disk."""
    ct = city_tile(city_name)
    tiles, _ = read_tile_pair(city_name, ct)
    tile_t1, tile_t5 = tiles["T1"], tiles["T5"]
    r0, c0, r1, c1 = ct["r0"], ct["c0"], ct["r1"], ct["c1"]
    w, h = c1 - c0, r1 - r0
    mask_t1 = valid_mask(tile_t1)
    mask_t5 = valid_mask(tile_t5)
    common = mask_t1 & mask_t5

    hh_res = ncc_shift(tile_t1["HHHH"], tile_t5["HHHH"], max_shift=max_shift, valid=common)
    vv_res = ncc_shift(tile_t1["VVVV"], tile_t5["VVVV"], max_shift=max_shift, valid=common)

    chosen_mode = mode
    if mode == "auto":
        mag = max(abs(hh_res["dr"]), abs(hh_res["dc"]))
        agree = abs(hh_res["dr"] - vv_res["dr"]) + abs(hh_res["dc"] - vv_res["dc"])
        chosen_mode = "blockwise" if (mag >= 0.3 or agree >= 0.4) else "global"

    shift_info = estimate_shift(tile_t1, tile_t5, common, mode=chosen_mode,
                                block=block, max_shift=max_shift)
    shift_info["hh"] = {k: (float(v) if isinstance(v, (float, np.floating, np.integer)) else v)
                        for k, v in hh_res.items()}
    shift_info["vv"] = {k: (float(v) if isinstance(v, (float, np.floating, np.integer)) else v)
                        for k, v in vv_res.items()}
    shift_info["city"] = city_name
    shift_info["tile"] = {"col": c0, "row": r0, "w": w, "h": h}
    shift_info["valid_common_frac"] = float(common.mean())
    return shift_info


def preprocess_tile(tile_t1, tile_t5, shift_info, eps=1e-6):
    """Terapkan koreksi registrasi + loading; kembalikan dict siap pakai."""
    mask_t1 = valid_mask(tile_t1)
    mask_t5 = valid_mask(tile_t5)
    corrected_t5, warped_ok, dr_map, dc_map = apply_correction(tile_t5, shift_info, mask_t5)
    final_mask = mask_t1 & warped_ok

    covs = [build_covariance(tile_t1),
            build_covariance(corrected_t5)]
    covs = diagonal_loading(covs, eps=eps)

    psd = [psd_stats(C, final_mask) for C in covs]
    deti = [determinant_imag_stats(C, final_mask) for C in covs]

    return {
        "tile_t1": tile_t1,
        "tile_t5": corrected_t5,
        "mask": final_mask,
        "covs": covs,
        "psd": psd,
        "det_imag": deti,
        "shift": shift_info,
        "eps": eps,
        "dr_map": dr_map,
        "dc_map": dc_map,
    }


def _geotransform(city_name):
    """Transform geografis tile pada grid overlap (piksel = 0.5 arcsec)."""
    ct = city_tile(city_name)
    dx = GRID["pixel_deg"]
    dy = GRID["pixel_deg"]
    west = GRID["west"] + ct["c0"] * dx
    north = GRID["north"] - ct["r0"] * dy
    return {"north": float(north), "west": float(west), "dx": float(dx), "dy": float(-dy)}


def save_preprocessed(city, proc, out_dir=None):
    """Simpan band T1/T5 terkoreksi + mask + shift field ke prep.npz/meta.json."""
    slug = city_slug(city)
    if out_dir is None:
        out_dir = output_tile_dir(city) / "prep"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for ds, tile in (("T1", proc["tile_t1"]), ("T5", proc["tile_t5"])):
        for band in ALL_BANDS:
            arr = np.asarray(tile[band])
            arrays[f"{ds}_{band}"] = arr.astype(
                np.complex64 if np.iscomplexobj(arr) else np.float32)
    arrays["mask"] = proc["mask"].astype(np.uint8)
    arrays["dr_map"] = np.asarray(proc["dr_map"], dtype=np.float32)
    arrays["dc_map"] = np.asarray(proc["dc_map"], dtype=np.float32)

    npz_path = out_dir / "prep.npz"
    np.savez_compressed(npz_path, **arrays)

    ct = city_tile(city)
    shift = proc["shift"]
    meta = {
        "city": city,
        "slug": slug,
        "location": "L1",
        "tile": {"r0": ct["r0"], "c0": ct["c0"], "r1": ct["r1"], "c1": ct["c1"],
                 "w": ct["c1"] - ct["c0"], "h": ct["r1"] - ct["r0"],
                 "lat": ct["lat"], "lon": ct["lon"]},
        "crops": {ds: list(v) for ds, v in ct["crops"].items()},
        "datasets": {ds: {"prefix": m["prefix"], "cx": m["cx"], "date": m["date"]}
                     for ds, m in DATASETS.items()},
        "geotransform": _geotransform(city),
        "params": {"p": POL_DIM, "m": LOOKS, "n": N_DATES, "eps": proc["eps"]},
        "registration": {
            "mode": shift["mode"],
            "global": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                       for k, v in shift["global"].items()},
            "hh": shift.get("hh"),
            "vv": shift.get("vv"),
            "grid_info": shift.get("grid_info"),
        },
        "valid_frac": float(proc["mask"].mean()),
        "arrays": {
            "T1": [f"T1_{b}" for b in ALL_BANDS],
            "T5": [f"T5_{b}" for b in ALL_BANDS],
            "mask": "mask", "dr_map": "dr_map", "dc_map": "dc_map",
        },
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path, meta_path


def run_phase1(cities=None, mode="auto", block=128, max_shift=8, eps=1e-6,
               out_path=None, save=False):
    """Estimasi koreksi registrasi untuk semua kota dan simpan laporan.

    Jika save=True, simpan band ter-preproses ke output/L1/<slug>/prep/.
    """
    if cities is None:
        cities = list(CITIES.keys())
    if out_path is None:
        out_path = PR_DIR / "fase1-preprocess.txt"

    lines = []
    lines.append("LAPORAN FASE 1 - PREPROCESSING & CO-REGISTRATION")
    lines.append("=" * 72)
    lines.append("")

    results = {}
    for city in cities:
        info = estimate_registration_city(city, mode=mode, block=block, max_shift=max_shift)
        hh = info["hh"]
        vv = info["vv"]
        lines.append(f"Kota: {city}")
        lines.append(f"  tile           : col={info['tile']['col']} row={info['tile']['row']} "
                     f"{info['tile']['w']}x{info['tile']['h']}")
        lines.append(f"  valid common   : {info['valid_common_frac']*100:.2f}%")
        lines.append(f"  HH NCC global  : dr={hh['dr']:+.4f} px, dc={hh['dc']:+.4f} px, r={hh['peak_r']:.4f}")
        lines.append(f"  VV NCC global  : dr={vv['dr']:+.4f} px, dc={vv['dc']:+.4f} px, r={vv['peak_r']:.4f}")
        lines.append(f"  mode dipakai   : {info['mode']}")
        if info["grid_info"]:
            gi = info["grid_info"]
            lines.append(f"  blockwise      : {gi['n_ok']}/{gi['n_total']} blok valid "
                         f"(block={gi['block']})")
            if gi["dr_ok_mean"] is not None:
                lines.append(f"    mean(dr,dc)  : ({gi['dr_ok_mean']:+.4f}, {gi['dc_ok_mean']:+.4f})")
        # dry-run preprocess untuk verifikasi PSD/det
        try:
            ct = city_tile(city)
            tiles, _ = read_tile_pair(city, ct)
            proc = preprocess_tile(tiles["T1"], tiles["T5"], info, eps=eps)
            lines.append(f"  mask akhir     : {proc['mask'].mean()*100:.2f}% valid")
            for ti, label in enumerate(["T1", "T5"]):
                psd = proc["psd"][ti]
                deti = proc["det_imag"][ti]
                n_nonpos = psd.get("n_nonpos", int(psd["n"] * (1 - psd.get("frac_psd", 1.0))))
                lines.append(
                    f"  {label} PSD: {psd['frac_psd']*100:.3f}% | min eig med {psd['min_eig_median']:.3e} "
                    f"| det imag max {deti['max_abs_imag']:.3e} (rel {deti['max_rel_imag']:.3e})"
                )
            before = info["valid_common_frac"]
            after = proc["mask"].mean()
            lines.append(f"  delta mask     : {(after-before)*100:+.3f} pp")
            if save:
                npz_path, meta_path = save_preprocessed(city, proc)
                lines.append(f"  disimpan       : {npz_path.relative_to(OUTPUT_DIR.parent)}")
                lines.append(f"                   {meta_path.relative_to(OUTPUT_DIR.parent)}")
        except Exception as e:
            lines.append(f"  preprocess dry-run: GAGAL ({e})")
        lines.append("")
        # simpan versi serializable
        results[city] = {k: v for k, v in info.items() if k != "grid"}

    lines.append("PARAMETER")
    lines.append("-" * 72)
    lines.append(f"  mode default   : auto (blockwise jika |shift|>=0.3 px atau HH-VV beda>=0.4)")
    lines.append(f"  block          : {block} px, max_shift {max_shift} px")
    lines.append(f"  diagonal load  : eps={eps} * trace(C)/p")
    lines.append(f"  interpolasi    : order=3 (bicubic) power; complex = real+imag terpisah")
    lines.append("  masker akhir   : valid(T1) & valid(T5 terkoreksi) & in-bounds")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_path.parent / "fase1-shifts.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    return out_path, results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fase 1: preprocessing + co-registration")
    ap.add_argument("--cities", nargs="*", default=None)
    ap.add_argument("--mode", choices=["auto", "global", "blockwise"], default="auto")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--max-shift", type=int, default=8)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--save", action="store_true", help="simpan prep.npz/meta.json")
    a = ap.parse_args()

    p, _ = run_phase1(cities=a.cities, mode=a.mode, block=a.block,
                      max_shift=a.max_shift, eps=a.eps, save=a.save)
    print(f"Hasil ditulis ke {p}")
    print(f"Ringkasan shift: {p.parent / 'fase1-shifts.json'}")
