import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    pafile = loadmat("../../acousticx/data/TEST.mat")['beamformed']
    print(pafile.shape)
    plt.figure(figsize = (15, 5))
    plt.imshow(pafile[100:, :, 100], cmap = "hot")
    plt.show()