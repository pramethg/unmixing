import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import *
from scipy.optimize import nnls
from sklearn.decomposition import FastICA

if __name__ == "__main__":
    wave_list = np.arange(700, 981, 10)
    # wave_list = [750, 760, 800, 825, 850, 880, 910, 925, 940, 950]
    legend = ["HbO2", "Hb", "Cholesterol", "Prostate"]
    wave_abs = np.load('./data/hbo2hbchpr_57.npy')
    abs_coeff = {}
    for idx, wave in enumerate(np.arange(700, 981, 5)):
        abs_coeff[wave] = (idx, wave_abs[idx])
    coeffs = np.vstack([abs_coeff[wave][1] for wave in wave_list])
    weights_plot(array = coeffs[:, 0:], wave_list = wave_list, legend = legend)

    hbhbo2fat = np.copy(coeffs)[:, 0:3]
    sim_data = np.array([np.array(loadmat(f"./data/hb_hbo2_fat_29_15/PA_Image_{wave}.mat")['Image_PA']) for wave in wave_list])
    unmixed = np.zeros((sim_data.shape[1], sim_data.shape[2], 3))
    for i in range(sim_data.shape[1]):
        for j in range(sim_data.shape[2]):
            unmixed[i, j] = nnls(hbhbo2fat, sim_data[:, i, j])[0]

    f = loadmat('./data/unmix.mat')
    X, Y = f['x'], f['y']
    clim = [0, 0.012]
    plot_3d(Y*1000, X*1000, unmixed[:, :, -1], title = legend[:3][-1], cmap = 'hot', clim = clim)
    plot_3d_multiple(Y*1000, X*1000, unmixed, title = legend[:3], cmap = 'jet', clim = clim)

    maps, wts = run_ica(sim_data, wave_list, 3, None)
    # plot_ica_2d(maps, wave_list, wts)
    # plot_3d_multiple(Y*1000, X*1000, maps, title = legend[:3], cmap = 'jet', clim = None, order = [1, 2, 0])