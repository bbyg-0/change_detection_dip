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
| T2 | 13 September 2023 | `eelriv_15807_23014_024_230913_PL09043020_05_CX_01` |

> **Catatan:** Repository ini **hanya menyertakan file annotation (`.ann`)** karena file data mentah berukuran sangat besar (total lebih dari 3 GB). Untuk menjalankan eksperimen, unduh data lengkap dari link di atas, lalu letakkan di folder `data/T1/` dan `data/T2/` sesuai struktur di bawah.

## Struktur Repository

```text
.
├── data/
│   ├── T1/
│   │   └── eelriv_15807_20037_012_201113_PL09043020_05_CX_02.ann
│   └── T2/
│       └── eelriv_15807_23014_024_230913_PL09043020_05_CX_01.ann
├── PR/
│   └── keputusan-teknis.txt
├── src/
│   ├── convert_uavsar.py
│   ├── to_geotiff.py
│   └── viewer.py
├── .gitignore
└── README.md
```

File data mentah yang perlu diunduh untuk tiap set (diletakkan di `data/T1/` atau `data/T2/`):

- `*HHHH_CX_*.grd`, `*HVHV_CX_*.grd`, `*VVVV_CX_*.grd` — power (real)
- `*HHHV_CX_*.grd`, `*HHVV_CX_*.grd`, `*HVVV_CX_*.grd` — cross-term (complex)
- `*_CX_*.inc` — sudut datang
- `*_CX_*.slope` — slope terrain
- `*_CX_*.ann` — annotation (sudah disertakan di repository)

## Instalasi

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy rasterio scipy
```

## Penggunaan

Konversi data UAVSAR ke GeoTIFF (opsional, untuk inspeksi):

```bash
python src/to_geotiff.py
```

Pipeline change detection (omnibus test statistic + K&I Weibull/gamma) akan ditambahkan pada tahap berikutnya.

## Catatan

- Folder `rujukan/` (berisi PDF paper) **tidak diunggah** ke repository.
- File data mentah (`.grd`, `.inc`, `.slope`, `.tif`, dsb.) diabaikan melalui `.gitignore`; hanya `.ann` yang dilacak.
- Keputusan teknis tahap awal tersimpan di `PR/keputusan-teknis.txt`.
