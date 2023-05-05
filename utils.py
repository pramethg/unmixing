import numpy as np
import matplotlib.pyplot as plt

def plot_3d(y, x, z, title = None, cmap = 'jet', clim = None, save = False):
    fig = plt.figure(figsize = (10, 10))
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

def plot_3d_multiple(y, x, z, title = None, cmap = 'jet', clim = None, save = False):
    fig = plt.figure(figsize = (18, 7))
    for i in range(z.shape[2]):
        ax = fig.add_subplot(1, z.shape[2], i + 1, projection='3d', proj_type = 'ortho')
        surf = ax.plot_surface(y, x, z[:, :, i], 
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

def plot_weights(array, legend = ["", "", ""], save = False, scale = False, div = 25, final = 951):
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