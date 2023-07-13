import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.decomposition import FastICA
import warnings
warnings.filterwarnings("ignore")

def normalize(image):
    mean = np.mean(image, axis = (0, 1))
    std = np.std(image, axis = (0, 1))
    normalized_image = (image - mean) / std
    scaled_image = (normalized_image - np.min(normalized_image)) / (np.max(normalized_image) - np.min(normalized_image))
    return scaled_image

def plot_3d(y, x, z, title = None, cmap = 'jet', clim = None, save = False):
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(projection='3d', proj_type = 'ortho')
    surf = ax.plot_surface(y, x, z, 
                            cmap = cmap, 
                            linewidth = 0, 
                            antialiased = False,
                            rstride = 1,
                            cstride = 1
                        )
    if clim is not None:
        surf.set_clim(clim[0], clim[1])
    fig.colorbar(surf, shrink = 0.6, aspect = 25, pad = -0.1)
    ax.grid(False)
    ax.view_init(azim = -90, elev = 90)
    ax.set_zticks([])
    plt.title(title, fontsize = 16)
    if save:
        plt.savefig(f"./{title}.png", dpi = 300, bbox_inches = 'tight')
    plt.show()

def plot_3d_multiple(y, x, z, title = None, cmap = 'jet', clim = None, save = False, order = [1, 2, 0]):
    fig = plt.figure(figsize = (15, 5))
    for i, j in zip(range(z.shape[2]), order):
        ax = fig.add_subplot(1, z.shape[2], i + 1, projection='3d', proj_type = 'ortho')
        surf = ax.plot_surface(y, x, z[:, :, j], 
                                cmap = cmap, 
                                linewidth = 0, 
                                antialiased = False,
                                rstride = 1,
                                cstride = 1
                            )
        if clim is not None:
            surf.set_clim(clim[0], clim[1])
        fig.colorbar(surf, shrink = 0.6, aspect = 25, pad = -0.1)
        ax.grid(False)
        ax.view_init(azim = -90, elev = 90)
        ax.set_zticks([])
        if title:
            ax.set_title(title[i], fontsize = 16)
    fig.tight_layout()
    if save:
        plt.savefig(f"./hbo2hbicg.png", dpi = 300, bbox_inches = 'tight')
    plt.show()

def wt_scale(array):
    minim, maxim = np.min(array), np.max(array)
    arr = (array - minim)/(maxim - minim)
    s = np.sum(arr, axis = 1)
    for i in range(array.shape[0]):
        arr[i]/=s[i]
    return arr

def plot_weights(array, legend = ["", "", ""], save = False, scale = False, div = 25, final = 981):
    plt.figure(figsize = (10, 6))
    plt.plot(np.arange(700, final, div), wt_scale(array) if scale else array)
    plt.xticks(np.arange(700, final, 20))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.legend(legend)
    plt.title(f"{legend[0]}, {legend[1]}, {legend[2]}")
    if save:
        plt.savefig(f"{legend[0]}{legend[1]}{legend[2]}.png")
    plt.show()

def weights_plot(array, wave_list, scale = False, legend = ["HbO2", "Hb", "Cholesterol", "Prostate"], save = False):
    plt.figure(figsize = (6, 4))
    plt.plot(wave_list, wt_scale(array) if scale else array)
    plt.xticks(wave_list)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.legend(legend)
    plt.title(f"{' '.join(legend)}: {len(wave_list)} Wavelengths")
    if save:
        plt.savefig(f"{''.join(legend)}".png)
    plt.show()

def run_ica(train_data, wave_list, n_components = 3, random_state = None):
    mdl = FastICA(n_components = n_components, algorithm = 'parallel', whiten = True, fun = 'exp', random_state = random_state)
    train_data = train_data.transpose((1, 2, 0)).reshape((-1, len(wave_list)))
    maps = mdl.fit_transform(train_data)
    ims = np.copy(maps).reshape((396, 101, 3))
    w = np.linalg.pinv(mdl.components_)
    return ims, w, mdl

def plot_comps_2d(comps, wave_list, wts, title = "ICA", figsize = (15, 4), order = [0, 1, 2], invert_sign = None, clim = None):
    ims = np.array([comps[:,:,i] for i in order]).transpose((1, 2, 0))
    w = np.array([wts[:,i] for i in order]).T
    chrom = ["HbO2", "Hb", "Cholesterol"]
    plt.figure(figsize = figsize)
    for i in range(ims.shape[2]):
        plt.subplot(1, 4, i+2)
        plt.imshow((-ims[:,:,i]) if invert_sign == i else (ims[:,:,i]), cmap = "hot")
        plt.title(chrom[i] + "(Inverted)" if invert_sign == i else chrom[i])
        if i == 2 and clim is not None:
            plt.clim(clim)
        plt.colorbar()
    plt.subplot(1, 4, 1)
    plt.plot(wave_list, w)
    plt.xticks(wave_list)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.title(label = title)
    plt.legend(['HbO2', 'Hb', 'Cholesterol'])
    plt.tight_layout()
    plt.show()

def run_linear_unmixing(sim_data, abscoeffs):
    unmixed = np.zeros((sim_data.shape[1], sim_data.shape[2], 3))
    for i in range(sim_data.shape[1]):
        for j in range(sim_data.shape[2]):
            unmixed[i, j] = nnls(abscoeffs, sim_data[:, i, j])[0]
    return unmixed