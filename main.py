import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from scipy.io import wavfile
from scipy.fft import fft, ifft, fftfreq

# ── Load the UI file ─────────────────────────────────────────────────────────
qtcreator_file = "design.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtcreator_file)


def fig_to_pixmap(filename="temp_plot.png"):
    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close('all')
    return QPixmap(filename)


class DesignWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(DesignWindow, self).__init__()
        self.setupUi(self)

        self.audio_signal = None
        self.audio_fe = None
        self.video_path = None

        # setScaledContents=True makes the pixmap always stretch to fill the label
        for lbl in [self.label_signal_temporel,
                    self.label_signal_echantillonne,
                    self.label_spectre,
                    self.label_video_preview]:
            lbl.setScaledContents(True)

        # Connections
        self.btn_load_wav.clicked.connect(self.handle_load_audio)
        self.btn_validate_resample.clicked.connect(self.handle_resampling)
        self.btn_compress_audio.clicked.connect(self.handle_audio_compression)
        self.btn_load_video.clicked.connect(self.handle_load_video)
        self.btn_compress_video.clicked.connect(self.handle_video_compression)

    # ════════════════════════════════════════════════════════════════════════
    # 4.1  AUDIO ANALYSIS
    # ════════════════════════════════════════════════════════════════════════

    def get_audio_info(self, filepath):
        fe, signal = wavfile.read(filepath)
        if signal.ndim == 1:
            n, c = len(signal), 1
            audio_type = "Mono"
        else:
            n, c = signal.shape
            audio_type = "Stéréo"
        duree = n / fe
        return {"fe": fe, "signal": signal, "type": audio_type,
                "n": n, "canaux": c, "duree": duree}

    def plot_temporal_to_pixmap(self, signal, fe, filename="temp_audio.png"):
        plt.ioff()
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 2))
        data = signal[:, 0] if signal.ndim > 1 else signal
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data.astype(np.float64) / max_val) * 1000
        ax.plot(data, linewidth=0.3, color='steelblue')
        ax.set_ylim(-1000, 1000)
        ax.set_yticks([-1000, 0, 1000])
        ax.set_title("Signal temporel", fontsize=8)
        ax.tick_params(labelsize=6)
        return fig_to_pixmap(filename)

    def handle_load_audio(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un fichier audio", "",
            "Fichiers audio (*.wav *.aiff *.mp3)")
        if not filepath:
            return
        try:
            info = self.get_audio_info(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de lire le fichier :\n{e}")
            return

        self.audio_fe = info["fe"]
        self.audio_signal = info["signal"]

        self.audio_features.setText(
            f"Type: {info['type']}\n"
            f"Fréquence: {info['fe']} Hz\n"
            f"Échantillons: {info['n']}\n"
            f"Durée: {info['duree']:.2f} s"
        )

        pixmap = self.plot_temporal_to_pixmap(self.audio_signal, self.audio_fe)
        self.label_signal_temporel.setPixmap(pixmap)

    # ════════════════════════════════════════════════════════════════════════
    # 4.2  RESAMPLING
    # ════════════════════════════════════════════════════════════════════════

    def resample_signal(self, signal, factor):
        if signal is None:
            return None
        data = signal[:, 0] if signal.ndim > 1 else signal
        return data[::factor]

    def plot_comparison_to_pixmap(self, original, resampled, factor):
        plt.ioff()
        plt.close('all')
        plt.figure(figsize=(6, 2))
        # Normalize both to [-500, 500] to match reference
        max_val = max(np.max(np.abs(original[:10000])), np.max(np.abs(resampled[:10000 // factor])))
        if max_val > 0:
            orig_norm = (original[:10000].astype(np.float64) / max_val) * 500
            res_norm  = (resampled[:10000 // factor].astype(np.float64) / max_val) * 500
        else:
            orig_norm = original[:10000]
            res_norm  = resampled[:10000 // factor]
        plt.plot(orig_norm, color='blue', linewidth=0.5, label="Original")
        indices = np.arange(len(res_norm)) * factor
        plt.plot(indices, res_norm, color='red', linewidth=0.4, label=f"Fe/{factor}")
        plt.ylim(-500, 500)
        plt.yticks([-500, 0, 500])
        plt.tick_params(labelsize=6)
        return fig_to_pixmap("temp_resample.png")

    def handle_resampling(self):
        if self.audio_signal is None:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord charger un fichier audio.")
            return

        factor = 2 if self.radio_fe2.isChecked() else (4 if self.radio_fe4.isChecked() else 8)
        data = self.audio_signal[:, 0] if self.audio_signal.ndim > 1 else self.audio_signal
        resampled = self.resample_signal(self.audio_signal, factor)

        pixmap = self.plot_comparison_to_pixmap(data, resampled, factor)
        self.label_signal_echantillonne.setPixmap(pixmap)

    # ════════════════════════════════════════════════════════════════════════
    # 4.3  AUDIO COMPRESSION
    # ════════════════════════════════════════════════════════════════════════

    def compress_audio_logic(self, signal_mono, r=128):
        fe = self.audio_fe
        z = fft(signal_mono.astype(np.float64))
        N = len(z)
        f = fftfreq(N, 1 / fe)

        modules = np.abs(z)
        modules_sorted = np.sort(modules)
        idx_seuil = int(N * (1 - 1 / r))
        seuil = modules_sorted[idx_seuil]

        z_compressed = z.copy()
        z_compressed[np.abs(z) < seuil] = 0

        return z_compressed, z, f

    def handle_audio_compression(self):
        if self.audio_signal is None:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord charger un fichier audio.")
            return

        data = self.audio_signal[:, 0] if self.audio_signal.ndim > 1 else self.audio_signal

        z_comp, z_orig, f = self.compress_audio_logic(data, r=128)

        # Plot raw FFT magnitude by sample index — matches reference exactly
        plt.ioff()
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 2))
        mag = np.abs(z_orig)
        ax.plot(mag, color='steelblue', linewidth=0.4)
        ax.set_title("Spectre du signal", fontsize=8)
        ax.tick_params(labelsize=6)

        pixmap = fig_to_pixmap("temp_spectre.png")
        self.label_spectre.setPixmap(pixmap)

    # ════════════════════════════════════════════════════════════════════════
    # 4.4  VIDEO ANALYSIS
    # ════════════════════════════════════════════════════════════════════════

    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.label_video_preview.setPixmap(QPixmap.fromImage(qt_img))

    def handle_load_video(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Ouvrir un fichier vidéo", "",
            "Fichiers vidéo (*.avi *.mp4 *.mov)")
        if not filepath:
            return

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            QMessageBox.critical(self, "Erreur", "Impossible d'ouvrir le fichier vidéo.")
            return

        fps       = cap.get(cv2.CAP_PROP_FPS)
        width     = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height    = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        nb_trames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        filename  = os.path.basename(filepath)

        self.video_path = filepath
        self.video_features.setText(
            f"Fichier: {filename}\n"
            f"Résolution: {int(width)}x{int(height)}\n"
            f"FPS: {fps:.2f}\n"
            f"Nombre de trames: {nb_trames:.2f}"
        )

        ret, frame = cap.read()
        if ret:
            self.display_frame(frame)
        cap.release()

    # ════════════════════════════════════════════════════════════════════════
    # 4.5  VIDEO COMPRESSION
    # ════════════════════════════════════════════════════════════════════════

    def handle_video_compression(self):
        if not self.video_path:
            QMessageBox.warning(self, "Attention", "Veuillez d'abord charger une vidéo.")
            return

        try:
            new_fps    = float(self.input_fps.text())
            new_width  = int(self.input_width.text())
            new_height = int(self.input_height.text())
        except ValueError:
            QMessageBox.critical(self, "Erreur", "FPS, Width et Height doivent être des nombres valides.")
            return

        selected_items = self.list_codec.selectedItems()
        codec_name = selected_items[0].text() if selected_items else "mp4v"

        ext_map = {"mp4v": ".mp4", "MJPG": ".avi", "XVID": ".avi"}
        ext = ext_map.get(codec_name, ".avi")
        os.makedirs("ressources", exist_ok=True)
        output_path = "ressources/output_compressed" + ext

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Erreur", "Impossible d'ouvrir la vidéo source.")
            return

        nb_trames    = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        orig_size_mb = os.path.getsize(self.video_path) / (1024 * 1024)

        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        out    = cv2.VideoWriter(output_path, fourcc, new_fps, (new_width, new_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(cv2.resize(frame, (new_width, new_height)))

        cap.release()
        out.release()

        new_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        self.video_results.setText(
            f"Ancienne taille: {orig_size_mb:.2f} MB\n"
            f"Nouvelle taille: {new_size_mb:.2f} MB\n"
            f"Résolution: {new_width}x{new_height}\n"
            f"FPS: {new_fps:.2f}\n"
            f"Nombre de trames: {nb_trames:.2f}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = DesignWindow()
    window.show()
    sys.exit(app.exec_())