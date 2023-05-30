import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import nnls
from utils import *
import pandas as pd
from sklearn.decomposition import FastICA

if __name__ == "__main__":
    num_chrom, num_wave, num_div, max_wave = 3, 29, 10, 981
    hbhbo2fat = np.load(f'./data/hbo2hbchpr_{num_wave}.npy')[:,0:num_chrom]
    plot_weights(hbhbo2fat, legend = ["HbO2", "Hb", "Cholesterol"], save = False, scale = False, div = num_div, final = max_wave)

    sim_data = np.array([np.array(loadmat(f"./data/hb_hbo2_fat_{num_wave}_15/PA_Image_{wave}.mat")['Image_PA']) for wave in np.arange(700, max_wave, num_div)])
    unmixed = np.zeros((sim_data.shape[1], sim_data.shape[2], num_chrom))
    for i in range(sim_data.shape[1]):
        for j in range(sim_data.shape[2]):
            unmixed[i, j] = nnls(hbhbo2fat, sim_data[:, i, j])[0]

    cart, title = False, ["HbO2", "Hb", "Cholesterol"]
    if cart:
        plt.figure(figsize = (8, 5))
        for wave in range(unmixed.shape[2]):
            plt.subplot(1, unmixed.shape[2], wave + 1).set_title(title[wave])
            plt.imshow(unmixed[:, :, wave], cmap = 'hot')
            plt.colorbar()
            plt.clim(0, 0.01)
            plt.axis('off')

    f = loadmat('./data/unmix.mat')
    X, Y = f['x'], f['y']
    clim = [0, 0.012]
    plot_3d(Y*1000, X*1000, unmixed[:, :, -1], title = title[-1], cmap = 'hot')#, clim = clim)
    plot_3d_multiple(Y*1000, X*1000, unmixed, title = title, cmap = 'jet')#, clim = clim)

    """
    mdl = FastICA(n_components = 3, algorithm = 'parallel', whiten = True, fun = 'exp', random_state = None)
    train_data = np.copy(sim_data)
    train_data = train_data.transpose((1, 2, 0)).reshape((-1, 29))
    maps = mdl.fit_transform(train_data)
    ims = np.copy(maps).reshape((396, 101, 3))
    w = mdl.components_.transpose()

    plt.figure(figsize = (15, 4))
    for i in range(3):
        plt.subplot(1, 4, i+2)
        plt.imshow(ims[:,:,i], cmap = "hot")
        plt.colorbar()
    plt.subplot(1, 4, 1)
    plt.plot(np.arange(700, 981, 10), w)
    plt.xticks(np.arange(700, 951, 25))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.title("ICA")
    plt.tight_layout()
    plt.show()
    """