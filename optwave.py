import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.io import loadmat
from utils import *
from scipy.linalg import svd

def selectwave(abscoeffs, threshold):
    pass

if __name__ == "__main__":
    wave_list = np.arange(700, 981, 10)
    legend = ["HbO2", "Hb", "Cholesterol", "Prostate"]
    wave_abs = np.load('./data/hbo2hbchpr_57.npy')
    abs_coeff = {}
    for idx, wave in enumerate(np.arange(700, 981, 5)):
        abs_coeff[wave] = (idx, wave_abs[idx])
    coeffs = np.vstack([abs_coeff[wave][1] for wave in wave_list])
    weights_plot(array = coeffs[:, 0:], wave_list = wave_list, legend = legend)
    abscoeffs = coeffs[:,0:3]