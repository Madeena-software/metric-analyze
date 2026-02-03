import os
import glob
import numpy as np
import pandas as pd
import tifffile
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import UnivariateSpline
from skimage import filters


class XRayQualityAnalyzer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.results = []

    def load_image(self, filepath):
        try:
            img = tifffile.imread(filepath)
            if img.ndim > 2:
                img = img[:, :, 0]
            return img.astype(np.float64)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def get_auto_rois(self, img):
        """
        Menggunakan Otsu Thresholding + Erosi untuk mendapatkan masker.
        Ditambahkan logika ekstra untuk memastikan background benar-benar area gelap.
        """
        # Normalisasi ke 8-bit
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Otsu thresholding
        try:
            val = filters.threshold_otsu(img_8bit)
            mask = img_8bit > val  # True = Objek (Signal)
        except:
            mask = np.zeros_like(img_8bit, dtype=bool)
            h, w = img.shape
            mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True

        # Erosi masker Signal (supaya tidak kena pinggiran)
        kernel = np.ones((9, 9), np.uint8)  # Kernel lebih besar agar lebih aman
        mask_signal = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)

        # Masker Background: Inversi dari masker objek, lalu di-erosi juga
        mask_bg = cv2.erode((~mask).astype(np.uint8), kernel, iterations=2)

        # --- LOGIKA TAMBAHAN UNTUK BACKGROUND ---
        # Seringkali background 'kotor'. Kita hanya ambil pixel background
        # yang intensitasnya di bawah rata-rata background itu sendiri (cari yang paling gelap/kosong)
        bg_candidates = img[mask_bg == 1]
        if len(bg_candidates) > 0:
            bg_threshold = np.percentile(bg_candidates, 50)  # Ambil 50% tergelap saja
            # Update mask_bg agar hanya mengambil area gelap
            mask_bg_refined = (mask_bg == 1) & (img < bg_threshold)
        else:
            mask_bg_refined = mask_bg == 1

        signal_pixels = img[mask_signal == 1]
        bg_pixels = img[mask_bg_refined]

        # Fallback
        if len(signal_pixels) == 0:
            h, w = img.shape
            signal_pixels = img[
                h // 2 - 50 : h // 2 + 50, w // 2 - 50 : w // 2 + 50
            ].flatten()

        if len(bg_pixels) < 100:
            # Ambil pojok kiri atas 100x100 pixel
            bg_pixels = img[:100, :100].flatten()
            # Update mask visualisasi (manual kotak pojok)
            mask_bg_refined = np.zeros_like(img, dtype=bool)
            mask_bg_refined[:100, :100] = True

        return signal_pixels, bg_pixels, mask_signal, mask_bg_refined

    def calculate_snr_cnr(self, signal_pixels, bg_pixels):
        mean_sig = np.mean(signal_pixels)
        mean_bg = np.mean(bg_pixels)
        std_bg = np.std(bg_pixels)

        if std_bg == 0:
            std_bg = 1e-6

        snr = mean_sig / std_bg
        cnr = np.abs(mean_sig - mean_bg) / std_bg
        return snr, cnr

    def calculate_mtf_fwhm(self, img):
        # (Sama seperti sebelumnya)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        grad = np.sqrt(sobelx**2 + sobely**2)
        _, _, _, max_loc = cv2.minMaxLoc(grad)
        x, y = max_loc
        roi_s = 30
        y1, y2 = max(0, y - roi_s), min(img.shape[0], y + roi_s)
        x1, x2 = max(0, x - roi_s), min(img.shape[1], x + roi_s)
        edge_roi = img[y1:y2, x1:x2]

        if edge_roi.size == 0:
            return 0, 0

        if abs(sobelx[y, x]) > abs(sobely[y, x]):
            esf = np.mean(edge_roi, axis=0)
        else:
            esf = np.mean(edge_roi, axis=1)

        lsf = np.abs(np.gradient(esf))

        # FWHM
        try:
            x_ax = np.arange(len(lsf))
            spl = UnivariateSpline(x_ax, lsf - (np.max(lsf) / 2), s=0)
            r = spl.roots()
            fwhm = abs(r[-1] - r[0]) if len(r) >= 2 else 0
        except:
            fwhm = 0

        # MTF
        try:
            mtf = np.abs(np.fft.fft(lsf))
            mtf = mtf[: len(mtf) // 2]
            mtf = mtf / (np.max(mtf) + 1e-9)
            idx = np.where(mtf < 0.5)[0]
            mtf50 = (idx[0] / len(mtf)) if len(idx) > 0 else 0
        except:
            mtf50 = 0

        return fwhm, mtf50

    def process_folder(self):
        exts = ["*.tif", "*.tiff"]
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.folder_path, e)))

        if not files:
            return None

        print(f"Memproses {len(files)} file...")

        for fp in files:
            img = self.load_image(fp)
            if img is None:
                continue

            # Ambil masks juga untuk keperluan debug nanti (optional)
            sig, bg, _, _ = self.get_auto_rois(img)

            snr, cnr = self.calculate_snr_cnr(sig, bg)
            fwhm, mtf = self.calculate_mtf_fwhm(img)

            self.results.append(
                {
                    "Filename": os.path.basename(fp),
                    "SNR": snr,
                    "CNR": cnr,
                    "FWHM": fwhm,
                    "MTF50": mtf,
                }
            )

        return pd.DataFrame(self.results)

    def visualize_roi_check(self, filename):
        """
        Fungsi KHUSUS untuk mengecek kenapa nilai SNR rendah.
        Menampilkan area mana yang dianggap SIGNAL dan BACKGROUND.
        """
        filepath = os.path.join(self.folder_path, filename)
        img = self.load_image(filepath)
        if img is None:
            return

        # Dapatkan ROI
        sig_px, bg_px, mask_sig, mask_bg = self.get_auto_rois(img)
        snr, cnr = self.calculate_snr_cnr(sig_px, bg_px)

        # Plotting
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(img, cmap="gray")
        ax[0].set_title("Citra Asli")

        # Visualisasi Masker Signal (Hijau)
        vis_sig = np.copy(img)
        # Buat overlay hijau
        overlay_sig = np.zeros((*img.shape, 3), dtype=np.uint8)
        overlay_sig[mask_sig == 1] = [0, 255, 0]  # Hijau
        ax[1].imshow(img, cmap="gray")
        ax[1].imshow(overlay_sig, alpha=0.3)
        ax[1].set_title(f"Area SIGNAL (Mean: {np.mean(sig_px):.1f})")

        # Visualisasi Masker Background (Merah)
        overlay_bg = np.zeros((*img.shape, 3), dtype=np.uint8)
        overlay_bg[mask_bg == 1] = [255, 0, 0]  # Merah
        ax[2].imshow(img, cmap="gray")
        ax[2].imshow(overlay_bg, alpha=0.3)
        ax[2].set_title(
            f"Area BACKGROUND (StdDev: {np.std(bg_px):.1f})\nSNR Result: {snr:.2f}"
        )

        plt.suptitle(f"Diagnosa ROI: {filename}", fontsize=14)
        plt.tight_layout()
        plt.show()


# ==========================================
#              MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # GANTI PATH FOLDER DI SINI
    folder_path = r"C:\Data_Penelitian\Citra_XRAY"

    if os.path.exists(folder_path):
        analyzer = XRayQualityAnalyzer(folder_path)
        df = analyzer.process_folder()

        if df is not None:
            # Urutkan berdasarkan SNR untuk melihat yang 'terbaik'
            df = df.sort_values(by="SNR", ascending=False)
            print("\n--- Hasil Analisis (Top 5 SNR) ---")
            print(df.head(5))

            # --- BAGIAN DIAGNOSA ---
            # Ambil file dengan SNR terbaik untuk divisualisasikan ROI-nya
            best_file = df.iloc[0]["Filename"]
            print(
                f"\n[DIAGNOSA] Memvisualisasikan area pengambilan sampel untuk: {best_file}"
            )
            print(
                "Perhatikan Area MERAH (Background). Jika area merah mengenai tulang/objek, SNR akan rendah."
            )
            analyzer.visualize_roi_check(best_file)

            # Simpan Excel
            df.to_excel(os.path.join(folder_path, "Hasil_Revisi_SNR.xlsx"), index=False)
