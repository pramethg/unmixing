import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    pafile = loadmat("../../acousticx/data/beamformed/11062023-ft_760_even_915_odd_co2_30_90_30_OBP_Laser_PA_1477.mat")['beamformed']
    plt.figure(figsize = (15, 5))
    plt.imshow(pafile[10, :, :], cmap = "hot")
    plt.show()