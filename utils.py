import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import warnings
warnings.filterwarnings("ignore")

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
    plt.figure(figsize = (8, 6))
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
    plt.figure(figsize = (8, 6))
    plt.plot(wave_list, wt_scale(array) if scale else array)
    plt.xticks(np.arange(min(wave_list), max(wave_list), 20))
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
    w = mdl.components_.transpose()
    return ims, w

def plot_ica_2d(ims, wave_list, w):
    plt.figure(figsize = (15, 4))
    for i in range(3):
        plt.subplot(1, 4, i+2)
        plt.imshow(ims[:,:,i], cmap = "hot")
        plt.colorbar()
    plt.subplot(1, 4, 1)
    plt.plot(wave_list, w)
    plt.xticks(np.arange(min(wave_list), max(wave_list), 20))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.title("ICA")
    plt.tight_layout()
    plt.show()