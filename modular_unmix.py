import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from utils import *
from scipy.optimize import nnls
from sklearn.decomposition import FastICA

def weights_plot(array, wave_list, scale = False, legend = ["HbO2", "Hb", "Cholesterol", "Prostate"], save = False):
    plt.figure(figsize = (10, 6))
    plt.plot(wave_list, wt_scale(array) if scale else array)
    # plt.xticks(np.linspace(min(wave_list), max(wave_list), 15))
    plt.xticks(wave_list)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.legend(legend)
    plt.title(f"{' '.join(legend)}: {len(wave_list)} Wavelengths")
    if save:
        plt.savefig(f"{''.join(legend)}".png)
    plt.show()

if __name__ == "__main__":
    # wave_list = np.arange(700, 981, 5)
    wave_list = [750, 760, 800, 825, 850, 880, 910, 925, 940, 950]
    legend = ["HbO2", "Hb", "Cholesterol", "Prostate"]
    wave_abs = np.load('./data/hbo2hbchpr_57.npy')
    abs_coeff = {}
    for idx, wave in enumerate(np.arange(700, 981, 5)):
        abs_coeff[wave] = (idx, wave_abs[idx])
    coeffs = np.vstack([abs_coeff[wave][1] for wave in wave_list])
    weights_plot(array = coeffs[:, 0:], wave_list = wave_list, legend = legend)

    hbhbo2fat = np.copy(coeffs)[:, 0:3]
    sim_data = np.array([np.array(loadmat(f"./data/hb_hbo2_fat_57/PA_Image_{wave}.mat")['Image_PA']) for wave in wave_list])
    unmixed = np.zeros((sim_data.shape[1], sim_data.shape[2], 3))
    for i in range(sim_data.shape[1]):
        for j in range(sim_data.shape[2]):
            unmixed[i, j] = nnls(hbhbo2fat, sim_data[:, i, j])[0]

    f = loadmat('./data/unmix.mat')
    X, Y = f['x'], f['y']
    clim = [0, 0.012]
    plot_3d(Y*1000, X*1000, unmixed[:, :, -1], title = legend[:3][-1], cmap = 'hot', clim = clim)
    plot_3d_multiple(Y*1000, X*1000, unmixed, title = legend[:3], cmap = 'jet', clim = clim)

    """
    mdl = FastICA(n_components = 3, algorithm = 'parallel', whiten = True, fun = 'exp', random_state = None)
    train_data = np.copy(sim_data)
    train_data = train_data.transpose((1, 2, 0)).reshape((-1, len(wave_list)))
    maps = mdl.fit_transform(train_data)
    ims = np.copy(maps).reshape((396, 101, 3))
    w = mdl.components_.transpose()

    plt.figure(figsize = (15, 4))
    for i in range(3):
        plt.subplot(1, 4, i+2)
        plt.imshow(ims[:,:,i], cmap = "hot")
        plt.colorbar()
    plt.subplot(1, 4, 1)
    plt.plot(wave_list, w)
    plt.xticks(wave_list)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.title("ICA")
    plt.tight_layout()
    plt.show()
    """