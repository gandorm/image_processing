import numpy as np
import matplotlib.pyplot as plt
from skimage import io


class Convolute:
    def __init__(self, file_path):
        img = io.imread(file_path)[:, :, 0]
        self.img = np.array(img, dtype=np.int)

    edge_detection_horizontal = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    edge_detection_vertical = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    edge_detection_gradient_magnitude1 = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])

    edge_detection_gradient_magnitude2 = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])

    def conv(self, kernel=edge_detection_horizontal):
        from scipy import signal

        temp = signal.convolve2d(self.img, kernel, mode='same')
        plt.axis('off')
        plt.imshow(temp, cmap='gray')
        plt.show()
