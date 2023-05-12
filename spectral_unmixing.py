import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import nnls
from utils import *
import pandas as pd
from sklearn.decomposition import FastICA

if __name__ == "__main__":
    hbhbo2fat = np.load('./data/hbo2hbchpr_57.npy')[:,0:3]
    plot_weights(hbhbo2fat, legend = ["HbO2", "Hb", "Cholesterol"], save = False, scale = False, div = 5, final = 981)

    sim_data = np.array([np.array(loadmat(f"./data/hb_hbo2_fat_57/PA_Image_{wave}.mat")['Image_PA']) for wave in np.arange(700, 981, 5)])
    unmixed = np.zeros((sim_data.shape[1], sim_data.shape[2], 3))
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
    plot_3d(Y*1000, X*1000, unmixed[:, :, 2], title = title[2], cmap = 'hot', clim = clim)
    plot_3d_multiple(Y*1000, X*1000, unmixed, title = title, cmap = 'jet', clim = clim)

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