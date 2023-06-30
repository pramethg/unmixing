import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.io import loadmat
from utils import *

if __name__ == "__main__":
    depth = 35
    # wave_list = np.arange(700, 981, 10)
    wave_list = [750, 760, 800, 830, 850, 880, 910, 920, 940, 950]
    legend = ["HbO2", "Hb", "Cholesterol", "Prostate"]
    wave_abs = np.load('./data/hbo2hbchpr_57.npy')
    abs_coeff = {}
    for idx, wave in enumerate(np.arange(700, 981, 5)):
        abs_coeff[wave] = (idx, wave_abs[idx])
    coeffs = np.vstack([abs_coeff[wave][1] for wave in wave_list])
    weights_plot(array = coeffs[:, 0:], wave_list = wave_list, legend = legend)

    hbhbo2fat = np.copy(coeffs)[:, 0:3]
    sim_data = np.array([np.array(loadmat(f"./data/hb_hbo2_fat_29_{depth}/PA_Image_{wave}.mat")['Image_PA']) for wave in wave_list])
    unmixed = run_linear_unmixing(sim_data, hbhbo2fat)

    f = loadmat('./data/unmix.mat')
    X, Y = f['x'], f['y']
    # clim = [0, 0.012]
    # plot_3d(Y*1000, X*1000, unmixed[:, :, -1], title = legend[:3][-1], cmap = 'hot', clim = clim)
    plot_3d_multiple(Y*1000, X*1000, unmixed, title = legend[:3], cmap = 'jet', clim = None, order = [0, 1, 2])

    maps, wts = run_ica(sim_data, wave_list, 3, None)
    plot_ica_2d(maps, wave_list, wts)
    # plot_3d_multiple(Y*1000, X*1000, maps, title = legend[:3], cmap = 'jet', clim = None, order = [1, 2, 0])