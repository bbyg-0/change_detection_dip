# Change Detection PolSAR UAVSAR P-Band Eel River

Eksperimen replikasi metode change detection berbasis **omnibus likelihood ratio test statistic** dan **Kittler & Illingworth (K&I) dengan distribusi Weibull/gamma**, mengikuti paper:

> Zhao, J., Chang, Y., Yang, J., Niu, Y., Lu, Z., & Li, P. (2020). *A Novel Change Detection Method Based on Statistical Distribution Characteristics Using Multi-Temporal PolSAR Data*. Sensors, 20(5), 1508. https://doi.org/10.3390/s20051508

Data yang dipakai adalah citra **UAVSAR P-band** (quad-pol, produk GRD) di area Eel River, California, pada dua tanggal berbeda.

## Sumber Data

Data diperoleh dari **UAVSAR JPL**:

**https://uavsar.jpl.nasa.gov/cgi-bin/data.pl**

Dataset yang digunakan:

| Set | Tanggal | Dataset ID |
| --- | --- | --- |
| T1 | 13 November 2020 | `eelriv_15807_20037_012_201113_PL09043020_05_CX_02` |
| T5 | 13 September 2023 | `eelriv_15807_23014_024_230913_PL09043020_05_CX_01` |

> **Catatan:** Repository ini **hanya menyertakan file annotation (`.ann`)** karena file data mentah berukuran sangat besar (total lebih dari 3 GB). Untuk menjalankan eksperimen, unduh data lengkap dari link di atas, lalu letakkan di folder `data/L1/T1/` dan `data/L1/T5/` sesuai struktur di bawah.

## Struktur Repository

```text
.
├── data/
│   └── L1/
│       ├── T1/
│       │   └── eelriv_15807_20037_012_201113_PL09043020_05_CX_02.ann
│       └── T5/
│           └── eelriv_15807_23014_024_230913_PL09043020_05_CX_01.ann
├── PR/
│   ├── keputusan-teknis.txt
│   ├── fase0-verifikasi.txt
│   ├── fase1-preprocess.txt
│   ├── fase2-parity.txt
│   └── fase2-hasil.txt
├── src/                      # I/O, preprocessing, ekspor (Python)
│   ├── config.py
│   ├── uavsar_io.py
│   ├── covariance.py
│   ├── registration.py
│   ├── preprocess.py
│   ├── fase0_verify.py
│   ├── omnibus_ref.py        # referensi parity DI
│   ├── ki_threshold.py       # referensi parity K&I
│   ├── export_results.py
│   ├── plot_results.py
│   ├── sampling.py
│   ├── patches.py
│   ├── optical.py
│   ├── optical_context.py
│   ├── labeling.py
│   ├── make_samples.py
│   ├── combine_labels.py
│   ├── binarize_di.py
│   ├── convert_uavsar.py
│   ├── to_geotiff.py
│   └── viewer.py
├── julia/                    # DI, K&I, baseline (Julia)
│   ├── Project.toml
│   └── src/
│       ├── UAVSARChangeDetection.jl
│       ├── prep.jl
│       ├── omnibus.jl
│       ├── ki.jl
│       ├── baselines.jl
│       ├── run_di.jl
│       ├── run_ki.jl
│       └── run_baselines.jl
├── output/                   # hasil (di-ignore git)
├── .gitignore
├── requirements.txt
└── README.md
```

File data mentah yang perlu diunduh untuk tiap set (diletakkan di `data/L1/T1/` atau `data/L1/T5/`):

- `*HHHH_CX_*.grd`, `*HVHV_CX_*.grd`, `*VVVV_CX_*.grd` — power (real)
- `*HHHV_CX_*.grd`, `*HHVV_CX_*.grd`, `*HVVV_CX_*.grd` — cross-term (complex)
- `*_CX_*.inc` — sudut datang
- `*_CX_*.slope` — slope terrain
- `*_CX_*.ann` — annotation (sudah disertakan di repository)

## Instalasi

Python (I/O, preprocessing, ekspor):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Julia (DI, K&I, baseline):

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

Dependensi Python:

| Paket | Kegunaan |
|---|---|
| `numpy`, `rasterio`, `scipy` | baca data GRD, geotransform, referensi K&I |
| `matplotlib`, `pillow` | visualisasi DI/peta perubahan dan patch citra |
| `shapely`, `pystac-client`, `planetary-computer` | ambil citra optik (NAIP, Sentinel-2) untuk ground truth |

Dependensi Julia: `NPZ`, `Distributions`, `StaticArrays`, `Roots`, `JSON3`, `SpecialFunctions`.

## Penggunaan

**1. Verifikasi data (Fase 0):**

```bash
python3 src/fase0_verify.py
```

**2. Preprocessing + co-registration + simpan data siap-DI (Fase 1.5):**

```bash
python src/preprocess.py --save
```

Menghasilkan `output/L1/<kota>/prep/prep.npz` + `meta.json` dan laporan `PR/fase1-preprocess.txt`.

**3. DI omnibus (Julia):**

```bash
julia --project=julia julia/src/run_di.jl \
  output/L1/arcata/prep/prep.npz output/L1/arcata/prep/meta.json \
  output/L1/arcata/di/arcata pos
```

**4. Threshold K&I (Julia):**

```bash
julia --project=julia julia/src/run_ki.jl \
  output/L1/arcata/di/arcata_di.npz \
  output/L1/arcata/change_map/arcata_gamma gamma 2048 likelihood 5
```

**5. Baseline (Julia):**

```bash
julia --project=julia julia/src/run_baselines.jl \
  output/L1/arcata/prep/prep.npz output/L1/arcata/prep/meta.json \
  output/L1/arcata/di/arcata_di.npz output/L1/arcata/change_map/arcata
```

**6. Ekspor GeoTIFF + figur (Python):**

```bash
python src/export_results.py
python src/plot_results.py
```

Hasil tersimpan di `output/L1/<kota>/` (`prep/`, `di/`, `change_map/`, `figures/`). Ringkasan metrik ada di `PR/fase2-hasil.txt`; parity Python–Julia di `PR/fase2-parity.txt`.

**7. Sampling, patch, optik, dan alat labeling (Fase 4):**

```bash
# semua kota (SAR + optik); optik bisa dilewati dengan --skip-optical
python src/make_samples.py

# satu kota saja
python src/make_samples.py --cities Arcata
```

Menghasilkan `output/L1/<kota>/samples/samples.csv`, `patches/*.png` (termasuk
konteks NAIP ~770 m), dan `labeling/labeling.html`. Buka HTML tersebut di browser (atau lewat
`python -m http.server -d output/L1/arcata 8000`), labeli dengan tombol
`1`/`0`/`u`/`s` (Spasi = crosshair on/off), lalu **Export CSV**.

Gabungkan CSV dari semua kota:

```bash
python src/combine_labels.py ~/Downloads/labels_*.csv --out labels/L1_labels.csv
```

Konversi data UAVSAR ke GeoTIFF (opsional, untuk inspeksi):

```bash
python src/to_geotiff.py
```

## Catatan

- Folder `rujukan/` (berisi PDF paper) **tidak diunggah** ke repository.
- File data mentah (`.grd`, `.inc`, `.slope`, `.tif`, dsb.) diabaikan melalui `.gitignore`; hanya `.ann` yang dilacak.
- Keputusan teknis tahap awal tersimpan di `PR/keputusan-teknis.txt`.
