from os import listdir
from os.path import isfile, join

from convolute import Convolute
from glcm import Glcm
from profile_histogram import Histogram

mypath = './img'


def print_summary(file_path):
    print(Glcm(file_path))
    Histogram(file_path).plot()
    Convolute(file_path).conv()


for f in listdir(mypath):
    if isfile(join(mypath, f)):
        print_summary(mypath + '/' + f)
