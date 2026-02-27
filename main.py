import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QFileDialog

# Importation de ton interface
from design import Ui_MainWindow

class ComputerVisionApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(ComputerVisionApp, self).__init__()
        self.setupUi(self)

        self.image = None
        self.processed_gray = None

        # --- Connexions des Boutons ---
        self.browseBtn.clicked.connect(self.charger_image)
        self.displayRChan.clicked.connect(lambda: self.extraire_canal(2)) # Rouge (Index 2 en BGR)
        self.displayGChan.clicked.connect(lambda: self.extraire_canal(1)) # Vert (Index 1 en BGR)
        self.displayBChan.clicked.connect(lambda: self.extraire_canal(0)) # Bleu (Index 0 en BGR)
        self.displayRChan.clicked.connect(lambda: self.extraire_canal(2))
        self.displayGChan.clicked.connect(lambda: self.extraire_canal(1))
        self.displayBChan.clicked.connect(lambda: self.extraire_canal(0))
        self.showHist.clicked.connect(self.generer_histogramme_couleur)

        self.DisplayGrayImg.clicked.connect(self.appliquer_transformation)
        self.DisplayGrayHist.clicked.connect(self.generer_histogramme_gris)

    def charger_image(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Ouvrir Image', '', 'Images (*.png *.jpg *.bmp)')
        if path:
            # Lecture robuste pour supporter les accents (Windows)
            stream = open(path, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            self.image = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

            if self.image is not None:
                self.afficher(self.image, self.originalImg)
                h, w, c = self.image.shape
                self.textBrowser.setText(f"Hauteur : {h}\nLargeur : {w}\nCanaux : {c}")

    def afficher(self, img, label):
        """Affiche une image OpenCV dans un QLabel PyQt5"""
        if len(img.shape) == 3:
            h, w, ch = img.shape
            # Conversion BGR vers RGB pour un affichage correct
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        else:
            h, w = img.shape
            qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio))

    def extraire_canal(self, canal_idx):
        if self.image is not None:
            res = np.zeros_like(self.image)
            res[:, :, canal_idx] = self.image[:, :, canal_idx]
            # Mapping des labels selon l'index
            labels = {2: self.rchan, 1: self.gchan, 0: self.bchan}
            self.afficher(res, labels[canal_idx])

    def generer_histogramme_couleur(self):
        if self.image is not None:
            plt.figure(figsize=(5, 3))
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
            plt.title("Histogramme BGR")
            plt.savefig("color_hist.png", bbox_inches='tight')
            plt.close()
            self.Hist.setPixmap(QPixmap("color_hist.png").scaled(self.Hist.width(), self.Hist.height()))

    def appliquer_transformation(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            try:
                a = float(self.contrast.text()) if self.contrast.text() else 1.0
                b = float(self.brightness.text()) if self.brightness.text() else 0.0
                self.processed_gray = cv2.convertScaleAbs(gray, alpha=a, beta=b)
                self.afficher(self.processed_gray, self.grayImg)
            except ValueError:
                print("Erreur de saisie")

    def generer_histogramme_gris(self):
        if self.processed_gray is not None:
            plt.figure(figsize=(5, 3))
            hist = cv2.calcHist([self.processed_gray], [0], None, [256], [0, 256])
            plt.plot(hist, color='black')
            plt.title("Histogramme Niveaux de Gris")
            plt.savefig("gray_hist.png", bbox_inches='tight')
            plt.close()
            self.grayHist.setPixmap(QPixmap("gray_hist.png").scaled(self.grayHist.width(), self.grayHist.height()))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = ComputerVisionApp()
    win.show()
    sys.exit(app.exec_())
    sys.exit(app.exec_())