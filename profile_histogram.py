import numpy as np
from skimage import io
import matplotlib.pyplot as plt


class Histogram:
    def __init__(self, file_path):
        img = io.imread(file_path)[:, :, 0]
        self.img = np.array(img, dtype=np.int)

    def profile_h(self):
        return np.sum(self.img, axis=0)

    def profile_v(self):
        return np.sum(self.img, axis=1)

    def plot(self):
        plt.plot(self.profile_h())
        plt.show()

        plt.plot(self.profile_v())
        plt.show()