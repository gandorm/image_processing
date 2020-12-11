import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from skimage.feature import greycomatrix, greycoprops
from skimage import data
from matplotlib import image
from matplotlib import pyplot
from skimage import io
import numpy as np

mypath = './img'


def profile_h(img):
    hprof = np.sum(img, axis=0)
    plt.plot(hprof)
    plt.show()

    return hprof


def profile_v(img):
    vprof = np.sum(img, axis=1)
    plt.plot()
    plt.show()

    return vprof


def calculate_asm(glcm):
    t = greycoprops(glcm, 'ASM')
    print(t)


def calculate_energy(glcm):
    ams = calculate_asm(glcm)
    return np.sqrt(ams)


def calculate_glcm_matrix(patch):
    return greycomatrix(patch, distances=[5], angles=[45], levels=256, symmetric=True, normed=True)


def conv(img):
    from scipy import signal

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

    # np.sqrt(temp1 ** 2 + temp2 ** 2)

    # for x in range(len(img[:, 1])):
    temp = signal.convolve2d(img, kernel, mode='same')
    plt.axis('off')
    plt.imshow(temp, cmap='gray')
    plt.show()

def read_img(file_path):
    img = io.imread(file_path)[:, :, 0]
    img = np.array(img, dtype=np.int)
    # thresh = threshold_otsu(img)
    # img = img > thresh
    cumulative_histogram(img)
    glcm = calculate_glcm_matrix(img)
    calculate_asm(glcm)
    conv(img)
    # summarize shape of the pixel array
    # print(img.dtype)
    # print(img.shape)
    # display the array of pixels as an image
    pyplot.imshow(img, cmap="gray")
    pyplot.show()

for f in listdir(mypath):
    if isfile(join(mypath, f)):
        print(f)
        read_img(mypath + '/' + f)
