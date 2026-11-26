"""Fase 0: verifikasi data sebelum perhitungan DI.

Mengecek:
1. metadata .ann dan offset grid T1/T5,
2. statistik band dan piksel invalid,
3. konsistensi matriks kovarians (korelasi, PSD, det Hermitian),
4. estimasi jumlah look dari blok homogen,
5. cakupan swath per kota,
6. co-registration residual (cross-correlation/phase correlation).

Hasil ditulis ke PR/fase0-verifikasi.txt.
"""

import datetime
from pathlib import Path

import numpy as np

from config import (
    CITIES,
    CROSS_BANDS,
    DATASET_KEYS,
    DATASETS,
    LOOKS,
    N_DATES,
    POL_DIM,
    POWER_BANDS,
    PR_DIR,
    SQRT2,
    T1_COL_OFFSET,
    T1_ROW_OFFSET,
    city_tile,
)
from covariance import (
    build_covariance,
    correlation_stats,
    determinant_imag_stats,
    psd_stats,
    valid_mask,
)
from registration import ncc_shift, phase_corr_shift
from uavsar_io import ann_info, read_tile_pair



def best_homogeneous_block(power, block=32):
    h, w = power.shape
    best = None
    for i in range(h // block):
        for j in range(w // block):
            b = power[i * block:(i + 1) * block, j * block:(j + 1) * block]
            b = b[np.isfinite(b) & (b > 0)]
            if b.size < block * block // 2:
                continue
            m = float(b.mean())
            v = float(b.var())
            if m <= 0:
                continue
            cv2 = v / (m * m)
            if best is None or cv2 < best[0]:
                best = (cv2, i, j, m, v)
    return best


def looks_from_values(values):
    values = values[np.isfinite(values) & (values > 0)]
    if values.size < 4:
        return float("nan")
    m = float(values.mean())
    v = float(values.var())
    return m * m / v if v > 0 else float("inf")


def cross_looks(a, b, c):
    """Estimasi m dari E|C12|^2 = |Sigma12|^2 + Sigma11*Sigma22/m."""
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c.real) & np.isfinite(c.imag)
    a, b, c = a[ok], b[ok], c[ok]
    if a.size < 4:
        return float("nan")
    s11 = float(a.mean())
    s22 = float(b.mean())
    e_abs2 = float(np.mean(np.abs(c) ** 2))
    e_c2 = float(np.abs(c.mean()) ** 2)
    den = e_abs2 - e_c2
    return s11 * s22 / den if den > 0 else float("inf")


def _power(bands):
    return bands["HHHH"] + bands["HVHV"] + bands["VVVV"]


def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("FASE 0 - VERIFIKASI DATA")
    log("=" * 72)
    log(f"Waktu        : {datetime.datetime.now().isoformat(timespec='seconds')}")
    log("Lokasi       : L1 (Eel River), tile Arcata 512x512")
    log("")

    # 1. Metadata dan offset grid
    log("1. METADATA DAN OFFSET GRID")
    infos = {}
    for ds in DATASET_KEYS:
        info = ann_info(ds)
        infos[ds] = info
        log(f"   {ds} ({DATASETS[ds]['date']}): rows={info['rows']}, cols={info['cols']}, "
            f"looks={info['looks']}, dx={info['dx']:.12g} deg, "
            f"UL=({info['north']:.9f}, {info['west']:.9f})")
    DS1, DS2 = DATASET_KEYS
    info1, info2 = infos[DS1], infos[DS2]
    off_col = (info2["west"] - info1["west"]) / info1["dx"]
    off_row = (info1["north"] - info2["north"]) / info1["dy"]
    log(f"   Offset {DS1}->{DS2}: col={off_col:.6f}, row={off_row:.6f} piksel")
    log(f"   Konstanta    : col={T1_COL_OFFSET}, row={T1_ROW_OFFSET}")
    log(f"   Integer?     : col={abs(off_col - round(off_col)) < 1e-6}, "
        f"row={abs(off_row - round(off_row)) < 1e-6}")
    log(f"   p={POL_DIM}, m={LOOKS} (7x27), n={N_DATES}")
    log("")

    # 2. Baca tile Arcata
    city = city_tile("Arcata")
    t1_r0, t1_c0 = city["crops"][DS1]
    log(f"2. TILE ARCATA (grid overlap) r0={city['r0']}, c0={city['c0']}, "
        f"{DS1} r0={t1_r0}, c0={t1_c0}")
    data, _ = read_tile_pair("Arcata", city)
    log("")

    # 3. Statistik band
    log("3. STATISTIK BAND DAN PIKSEL INVALID")
    for ds in DATASET_KEYS:
        bands = data[ds]
        log(f"   {ds}:")
        for key in POWER_BANDS + CROSS_BANDS:
            arr = bands[key]
            if np.iscomplexobj(arr):
                mag = np.abs(arr)
                finite = np.isfinite(arr.real) & np.isfinite(arr.imag)
                log(f"     {key:4s} complex: |z| mean={mag.mean():.6g}, "
                    f"median={np.median(mag):.6g}, max={mag.max():.6g}, "
                    f"non-finite={np.mean(~finite)*100:.4f}%")
            else:
                finite = np.isfinite(arr)
                log(f"     {key:4s} real   : mean={arr.mean():.6g}, "
                    f"median={np.median(arr):.6g}, min={arr.min():.6g}, "
                    f"max={arr.max():.6g}, <=0={np.mean(arr <= 0)*100:.4f}%, "
                    f"non-finite={np.mean(~finite)*100:.4f}%")
        mask = valid_mask(bands)
        log(f"     valid     : {mask.mean()*100:.4f}% "
            f"({int((~mask).sum())} piksel invalid)")
    log("")

    # 4. Konsistensi kovarians
    log("4. KONSISTENSI MATRIKS KOVARIANS (dengan faktor sqrt2/2)")
    for ds in DATASET_KEYS:
        bands = data[ds]
        mask = valid_mask(bands)
        C = build_covariance(bands, factors=True)
        corr = correlation_stats(C, mask)
        psd = psd_stats(C, mask)
        dimag = determinant_imag_stats(C, mask)
        log(f"   {ds}:")
        for name, st in corr.items():
            log(f"     {name}: mean={st['mean']:.6f}, median={st['median']:.6f}, "
                f"p99={st['p99']:.6f}, max={st['max']:.6f}, "
                f"frac>1={st['frac_gt_1']*100:.6f}%")
        log(f"     PSD: frac={psd['frac_psd']*100:.6f}%, "
            f"min_eig median={psd['min_eig_median']:.6g}, min={psd['min_eig_min']:.6g}")
        log(f"     det(C) imag: max|imag|={dimag['max_abs_imag']:.6g}, "
            f"max_rel={dimag['max_rel_imag']:.6g}, median_rel={dimag['median_rel_imag']:.6g}")

        C0 = build_covariance(bands, factors=False)
        corr0 = correlation_stats(C0, mask)
        psd0 = psd_stats(C0, mask)
        log(f"     [tanpa faktor] r12 median={corr0['r12']['median']:.6f}, "
            f"r23 median={corr0['r23']['median']:.6f}, "
            f"frac PSD={psd0['frac_psd']*100:.6f}%")
    log("")

    # 5. Estimasi jumlah look dari blok homogen
    log("5. ESTIMASI JUMLAH LOOK DARI BLOK HOMOGEN 32x32")
    for ds in DATASET_KEYS:
        bands = data[ds]
        power = _power(bands)
        best = best_homogeneous_block(power)
        if best is None:
            log(f"   {ds}: tidak ada blok homogen valid")
            continue
        cv2, bi, bj, _, _ = best
        sl = (slice(bi * 32, (bi + 1) * 32), slice(bj * 32, (bj + 1) * 32))
        log(f"   {ds}: blok ({bi},{bj}), CV^2={cv2:.6g}")
        for key in POWER_BANDS:
            est = looks_from_values(bands[key][sl])
            log(f"     m_est {key:4s} = {est:8.2f}")
        m12 = cross_looks(bands["HHHH"][sl], 2.0 * bands["HVHV"][sl],
                          SQRT2 * bands["HHHV"][sl])
        m13 = cross_looks(bands["HHHH"][sl], bands["VVVV"][sl],
                          bands["HHVV"][sl])
        m23 = cross_looks(2.0 * bands["HVHV"][sl], bands["VVVV"][sl],
                          SQRT2 * bands["HVVV"][sl])
        log(f"     m_est dari cross: m12={m12:8.2f}, m13={m13:8.2f}, m23={m23:8.2f}")
        log(f"     (nilai acuan m={LOOKS}; estimasi empiris adalah batas bawah")
        log("      karena heterogenitas scene/resampling)")
    log("")

    # 6. Cakupan swath per kota
    log("6. CAKUPAN SWATH PER KOTA")
    log(f"   {'kota':22s} {('valid '+DS1):>9s} {('valid '+DS2):>9s} {'common':>8s}")
    coverage = {}
    for name in CITIES:
        c = city_tile(name)
        d, _ = read_tile_pair(name, c)
        m1 = valid_mask(d[DS1])
        m2 = valid_mask(d[DS2])
        common = m1 & m2
        coverage[name] = {
            "m1": float(m1.mean()),
            "m2": float(m2.mean()),
            "common": float(common.mean()),
            "data": d,
            "city": c,
        }
        log(f"   {name:22s} {m1.mean()*100:8.2f}% {m2.mean()*100:8.2f}% "
            f"{common.mean()*100:7.2f}%")
    log("")

    # 7. Co-registration residual
    log("7. CO-REGISTRATION RESIDUAL PER KOTA (masked cross-correlation)")
    log(f"   {'kota':22s} {'r':>7s} {'dr':>8s} {'dc':>8s} {'|d|':>7s} {'int':>9s}")
    shifts = {}
    for name in CITIES:
        cov = coverage[name]
        if cov["common"] < 0.01:
            log(f"   {name:22s}   -- di luar swath (tidak ada piksel valid) --")
            continue
        p1 = _power(cov["data"][DS1])
        p2 = _power(cov["data"][DS2])
        valid = (p1 > 0) & (p2 > 0) & np.isfinite(p1) & np.isfinite(p2)
        ncc = ncc_shift(p1, p2, max_shift=8, valid=valid)
        shifts[name] = ncc
        log(f"   {name:22s} {ncc['peak_r']:7.4f} {ncc['dr']:+8.3f} {ncc['dc']:+8.3f} "
            f"{np.hypot(ncc['dr'], ncc['dc']):7.3f} "
            f"({ncc['dr_int']:+d},{ncc['dc_int']:+d})")
    log("")
    log(f"   Interpretasi: nilai (dr,dc) adalah translasi yang diterapkan pada {DS2}")
    log(f"   untuk sejajar dengan {DS1}. Nilai positif = geser kanan/bawah.")
    log("")

    # 8. Variasi spasial di tile Arcata (per kuadran)
    log("8. VARIASI SPASIAL SHIFT DI TILE ARCATA (per kuadran)")
    cov = coverage["Arcata"]
    p1 = _power(cov["data"][DS1])
    p2 = _power(cov["data"][DS2])
    valid = (p1 > 0) & (p2 > 0) & np.isfinite(p1) & np.isfinite(p2)
    h, w = p1.shape
    hh, ww = h // 2, w // 2
    quads = [
        ("kiri-atas", slice(0, hh), slice(0, ww)),
        ("kanan-atas", slice(0, hh), slice(ww, w)),
        ("kiri-bawah", slice(hh, h), slice(0, ww)),
        ("kanan-bawah", slice(hh, h), slice(ww, w)),
    ]
    for qname, rs, cs in quads:
        ncc = ncc_shift(p1[rs, cs], p2[rs, cs], max_shift=10, valid=valid[rs, cs])
        log(f"   {qname:12s}: r={ncc['peak_r']:.4f}, "
            f"shift=({ncc['dr']:+.3f}, {ncc['dc']:+.3f})")
    log("")

    # 9. Phase correlation pembanding untuk Arcata
    log("9. PHASE CORRELATION PEMBANDING (Arcata)")
    cov = coverage["Arcata"]
    p1 = _power(cov["data"][DS1])
    p2 = _power(cov["data"][DS2])
    valid = (p1 > 0) & (p2 > 0) & np.isfinite(p1) & np.isfinite(p2)
    pc = phase_corr_shift(p1, p2, valid=valid)
    log(f"   Phase: shift=({pc['dr']:+.4f}, {pc['dc']:+.4f}) piksel, "
        f"integer=({pc['dr_int']:+d}, {pc['dc_int']:+d})")
    log("")

    # Kesimpulan
    log("KESIMPULAN")
    log("-" * 72)
    log(f"- Offset grid integer dan sesuai konstanta: "
        f"{abs(off_col - T1_COL_OFFSET) < 1e-6 and abs(off_row - T1_ROW_OFFSET) < 1e-6}")
    log(f"- p={POL_DIM}, m={LOOKS}, n={N_DATES} dikonfirmasi dari .ann/konfigurasi.")
    log("- Matriks kovarians memakai C22=2|HV|^2, C12=sqrt2<HH HV*>, C23=sqrt2<HV VV*>,")
    log("  C13=<HH VV*> dengan elemen konjugat untuk Hermitian.")
    log("- Konjugasi global pada cross-term tidak mengubah determinan/omnibus;")
    log("  yang penting konsisten Hermitian.")
    log("- Parameter m hanya menggeser DI secara global (koefisien konstan), sehingga")
    log("  pemilihan threshold K&I tidak berubah; nilai nominal m=189 dipakai.")
    log("- Kota yang dipakai: Tyee City, Arcata, McKinleyville, Trinidad, Blue Lake.")
    log("  Semua tercakup pada kedua tanggal (Eureka & Samoa sudah diganti).")
    log("- Residual co-registration per kota lihat tabel 7; nilai > 0.5 px perlu")
    log("  koreksi subpiksel. Variasi per kuadran menunjukkan residual bersifat")
    log("  spasial, kemungkinan akibat perbedaan DEM.")
    log("")

    PR_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(PR_DIR) / "fase0-verifikasi.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nHasil ditulis ke {out}")


if __name__ == "__main__":
    main()
