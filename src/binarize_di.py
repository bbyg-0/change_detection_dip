
import numpy as np
from PIL import Image
from pathlib import Path


def main():
    print("=== Binary DI Image Generator ===")

    # 1. Input file NPY
    file_path = input("Masukkan path file DI (.npy): ").strip()
    file_path = Path(file_path.strip('"').strip("'"))

    if not file_path.is_file():
        print("Error: File tidak ditemukan.")
        return

    if file_path.suffix.lower() != ".npy":
        print("Error: File harus berformat .npy.")
        return

    try:
        di = np.load(file_path)
    except Exception as e:
        print(f"Error saat membaca file: {e}")
        return

    # 2. Validasi citra
    if di.ndim != 2:
        print(f"Error: DI harus berupa citra 2D. Shape: {di.shape}")
        return

    if not np.issubdtype(di.dtype, np.number):
        print("Error: Data DI harus berupa numerik.")
        return

    if not np.isfinite(di).all():
        print("Error: DI mengandung NaN atau infinity.")
        return

    # 3. Input threshold dalam rentang 0-255
    try:
        threshold = float(input("Masukkan threshold (0-255): "))
    except ValueError:
        print("Error: Threshold harus berupa angka.")
        return

    if not 0 <= threshold <= 255:
        print("Error: Threshold harus berada pada rentang 0-255.")
        return

    # 4. Konversi threshold ke skala DI asli
    threshold_di = threshold / 255.0

    # 5. Binarisasi
    # 1 = changed (putih)
    # 0 = unchanged (hitam)
    binary = (di > threshold_di).astype(np.uint8)

    # 6. Tentukan nama file output
    threshold_label = f"{threshold:g}".replace(".", "p")

    output_npy = file_path.with_name(
        f"{file_path.stem}_binary_T{threshold_label}.npy"
    )

    output_png = file_path.with_name(
        f"{file_path.stem}_binary_T{threshold_label}.png"
    )

    # 7. Simpan NPY (nilai 0 dan 1)
    np.save(output_npy, binary)

    # 8. Simpan PNG (nilai 0 dan 255)
    Image.fromarray(binary * 255).save(output_png)

    # 9. Tampilkan ringkasan
    total_pixels = binary.size
    changed = int(np.sum(binary))
    unchanged = total_pixels - changed

    print("\n=== Hasil Binarisasi ===")
    print(f"Ukuran DI          : {di.shape}")
    print(f"Rentang DI asli    : {di.min():.6f} - {di.max():.6f}")
    print(f"Threshold input    : {threshold:.4f} / 255")
    print(f"Threshold DI       : {threshold_di:.8f}")
    print(f"Piksel changed     : {changed}")
    print(f"Piksel unchanged   : {unchanged}")
    print(f"Persentase changed : {changed / total_pixels * 100:.2f}%")
    print(f"Output NPY         : {output_npy}")
    print(f"Output PNG         : {output_png}")


if __name__ == "__main__":
    main()

