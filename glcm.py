from skimage.feature import greycomatrix, greycoprops
from skimage import io
import numpy as np


class Glcm:
    def __init__(self, file_path):
        img = io.imread(file_path)[:, :, 0]
        self.img = np.array(img, dtype=np.int)
        self.glcm = self.calculate_glcm_matrix()

    def calculate_glcm_matrix(self):
        return greycomatrix(self.img, distances=[5], angles=[45], levels=256, symmetric=True, normed=True)

    def calculate_asm(self):
        return greycoprops(self.glcm, 'ASM')

    def calculate_contrast(self):
        return greycoprops(self.glcm, 'contrast')

    def calculate_dissimilarity(self):
        return greycoprops(self.glcm, 'dissimilarity')

    def calculate_homogeneity(self):
        return greycoprops(self.glcm, 'homogeneity')

    def calculate_correlation(self):
        return greycoprops(self.glcm, 'correlation')

    def calculate_energy(self):
        ams = self.calculate_asm()
        return np.sqrt(ams)

    def __str__(self):
        return "ASM: {}, Energy: {}, Contrast: {}, Correlation: {}, Dissimilarity: {}, Homogeneity: {}"\
            .format(
                str(self.calculate_asm()),
                str(self.calculate_energy()),
                str(self.calculate_contrast()),
                str(self.calculate_correlation()),
                str(self.calculate_dissimilarity()),
                str(self.calculate_homogeneity())
        )
