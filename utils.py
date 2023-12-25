import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import nnls
from sklearn.decomposition import FastICA
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings("ignore")

def normalize(image, axis = (1, 2), keepdims = True):
    minimg = np.min(image, axis = axis, keepdims = keepdims)
    maximg = np.max(image, axis = axis, keepdims = keepdims)
    normalized_image = (image - minimg) / (maximg - minimg)
    return normalized_image

def standardize(image, axis = (1, 2), keepdims = True):
    mean = np.mean(image, axis = axis, keepdims = keepdims)
    std = np.std(image, axis = axis, keepdims = keepdims)
    normalized_image = (image - mean) / std
    return normalized_image

def whiten(X):
    c, h, w = X.shape
    XW = X.reshape((c, h*w))
    X_mean = XW.mean(axis = -1)
    XW -= X_mean[:, np.newaxis]
    U, D = linalg.svd(XW, full_matrices = False, check_finite = False)[:2]
    U *= np.sign(U[0])
    K = (U / D).T
    XW = np.dot(K, XW)
    XW *= np.sqrt(h*w)
    XW = XW.reshape((c, h, w))
    return XW

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

def plot_3d_multiple(y, x, z, title = None, cmap = 'jet', clim = [None]*3, save = False, order = [1, 2, 0]):
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
        if clim[i] is not None:
            surf.set_clim(clim[i][0], clim[i][1])
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
    plt.ylabel("Absorption Coefficient (mm^-1)")
    plt.legend(legend)
    plt.title(f"{legend[0]}, {legend[1]}, {legend[2]}")
    if save:
        plt.savefig(f"{legend[0]}{legend[1]}{legend[2]}.png")
    plt.show()

def weights_plot(array, wave_list, scale = False, legend = ["HbO2", "Hb", "Cholesterol", "Prostate"], save = False, figsize = (6, 4), xticks = None, title = None):
    plt.figure(figsize = figsize)
    plt.plot(wave_list, wt_scale(array) if scale else array)
    plt.xticks(xticks)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (mm^-1)")
    plt.legend(legend)
    if title is None:
        plt.title(f"{' '.join(legend)}: {len(wave_list)} Wavelengths")
    else:
        plt.title(label = title)
    if save:
        plt.savefig(f"{''.join(legend)}".png)
    plt.show()

def run_ica(train_data, wave_list, n_components = 3, random_state = None, fun = 'exp', algorithm = 'parallel'):
    _, h, w = train_data.shape
    mdl = FastICA(n_components = n_components, algorithm = algorithm, whiten = True, fun = fun, random_state = random_state)
    train_data = train_data.transpose((1, 2, 0)).reshape((-1, len(wave_list)))
    maps = mdl.fit_transform(train_data)
    ims = np.copy(maps).reshape((h, w, n_components))
    w = np.linalg.pinv(mdl.components_)
    return ims, w, mdl

def plot_comps_2d(comps, wave_list, wts, title = "ICA", figsize = (15, 4), order = [0, 1, 2], invert_sign = None, clim = [None]*3, xticks = None, chrom = ['HbO2', 'Hb', 'Cholesterol'], save = None, mrows = None):
    ims = np.array([comps[:,:,i] for i in order]).transpose((1, 2, 0))
    if len(chrom) != ims.shape[2]:
        chrom = [str(i) for i in range(ims.shape[2])]
    w = np.array([wts[:,i] for i in order]).T
    plt.figure(figsize = figsize)
    if mrows:
        for i in range((mrows[0] * mrows[1]) - 1):
            if i >= ims.shape[-1]:
                break
            plt.subplot(mrows[0], mrows[1], i + 2)
            plt.imshow((-ims[:,:,i]) if invert_sign == i else (ims[:,:,i]), cmap = "hot")
            if chrom:
                plt.title(chrom[i] + "(Inverted)" if invert_sign == i else chrom[i])
            if clim[i] is not None:
                plt.clim(clim[i])
            plt.colorbar()
        plt.subplot(mrows[0], mrows[1], 1)
        plt.plot(wave_list if len(wave_list) == w.shape[0] else list(range(w.shape[0])), w)
        plt.xticks(xticks)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absorption Coefficient (mm^-1)")
        plt.title(label = title)
    else:
        for i in range(ims.shape[2]):
            plt.subplot(1, ims.shape[2] + 1, i+2)
            plt.imshow((-ims[:,:,i]) if invert_sign == i else (ims[:,:,i]), cmap = "hot")
            if chrom:
                plt.title(chrom[i] + "(Inverted)" if invert_sign == i else chrom[i])
            if clim[i] is not None:
                plt.clim(clim[i])
            plt.colorbar()
        plt.subplot(1, ims.shape[2] + 1, 1)
        plt.plot(wave_list if len(wave_list) == w.shape[0] else list(range(w.shape[0])), w)
        plt.xticks(xticks)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absorption Coefficient (mm^-1)")
        plt.title(label = title)
    plt.legend(chrom)
    plt.tight_layout()
    if save:
        plt.savefig(f'{save}.png', dpi = 500)
    else:
        plt.show()
    plt.close()

def run_linear_unmixing(sim_data, abscoeffs):
    unmixed = np.zeros((sim_data.shape[1], sim_data.shape[2], abscoeffs.shape[1]))
    for i in range(sim_data.shape[1]):
        for j in range(sim_data.shape[2]):
            unmixed[i, j] = nnls(abscoeffs, sim_data[:, i, j])[0]
    return unmixed

def roi_analysis(exp_img, img, rois = np.array([[[0, 10], [0, 10]]])):
    plt.figure(figsize = (7, 7))
    plt.yticks(np.arange(0, np.shape(exp_img)[1], 5))
    plt.xticks(np.arange(0, np.shape(exp_img)[2], 5))
    plt.imshow(img, cmap = 'hot')
    plt.colorbar()
    for idx in range(len(rois)):
        plt.gca().add_patch(Rectangle((rois[idx][1, 0], rois[idx][0, 0]), rois[idx][1, 1]-rois[idx][1, 0], rois[idx][0, 1]-rois[idx][0, 0],edgecolor = 'cyan', facecolor = 'None', lw = 1.5))
    plt.show()
    plt.close()
    plt.figure(figsize = (6 * len(rois), 4))
    for idx in range(len(rois)):
        plt.subplot(1, len(rois), idx + 1)
        plt.plot(wave_list, np.mean(exp_img[:, rois[idx][0, 0]:rois[idx][0, 1] + 1, rois[idx][1, 0]:rois[idx][1, 1] + 1], axis = (1, 2)))
        plt.title(f'ROI - {idx + 1}')
    plt.show()

def textparse(fpath = "../acousticx/TEST-5.txt"):
    res = []
    with open(fpath, 'r') as f:
        for line in f:
            roi = list(map(int, line.strip().split()))
            res.append(roi)
    return res

def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size

def roiplot(ax, img, rois, title):
    ax.imshow(img, cmap = 'hot')
    ax.set_title(f"{title}")
    for ridx in range(len(rois)):
        rect = Rectangle((rois[ridx][2], rois[ridx][0]), (rois[ridx][3] - rois[ridx][2]), (rois[ridx][1] - rois[ridx][0]), linewidth = 1, edgecolor = 'white', facecolor = 'none')
        ax.add_patch(rect)

def singleroi(pa, file, lidx):
    roi = textparse(os.path.join(dir, file))[lidx]
    print(roi)
    plt.figure(figsize = (8, ))
    plt.subplot(1, 2, 1)
    plt.imshow(pa[:, :, 9].T, cmap = "hot")
    plt.gca().add_patch(Rectangle((roi[2], roi[0]), roi[3] - roi[2], roi[1] - roi[0], edgecolor = 'white', facecolor = 'None', lw = 2.0))
    plt.subplot(1, 2, 2)
    mean = np.mean(pa[:, roi[0]:roi[1], roi[2]:roi[3]], axis = (1, 2))
    mavg = moving_average(mean, 5)
    plt.plot(mean[:-1], "b")
    plt.plot(mavg[:-1], 'r')
    plt.axvspan(round(pa.shape[0] * ratios[1][0] / 2), round(frames * ratios[1][1] / 2), color = 'coral', alpha = 0.4, lw = 0)

def singleplot(pa, rois, fidx, idx = 9):
    plt.figure(figsize = (7, 7))
    plt.imshow(pa[fidx, rois[idx][0]:rois[idx][1], rois[idx][2]:rois[idx][3]], cmap = 'hot', extent = [0, 1, 0, 1])
    plt.colorbar()
    plt.show()

"""
fig, axs = plt.subplots(2, 5, figsize = (25, 8))
roiplot(axs[0, 0], paw1[0, rois[8][0]: rois[8][1], rois[8][2]: rois[8][3]], rois[: 4], title = f"{swv[0]} ROIs")
roiplot(axs[1, 0], paw2[0, rois[8][0]: rois[8][1], rois[8][2]: rois[8][3]], rois[4:8], title = f"{swv[1]} ROIs")
for idx, subidx in enumerate(list(set(list(range(10))) - set([0, 5]))):
    ax = axs[subidx // 5, subidx % 5]
    if idx < nrois:
        mean = np.mean(paw1[:, rois[idx][0]:rois[idx][1], rois[idx][2]:rois[idx][3]], axis=(1, 2))
    else:
        mean = np.mean(paw2[:, rois[idx][0]:rois[idx][1], rois[idx][2]:rois[idx][3]], axis=(1, 2))
    mavg = moving_average(mean, 5)
    ax.plot(mean, "b")
    ax.plot(mavg, 'r')
    ax.axvspan(round(frames * ratios[1][0] / 2), round(frames * ratios[1][1] / 2), color='coral', alpha=0.4, lw=0)
    ax.set_title(f'{swv[0] if idx < nrois else swv[1]} NM ROI - {(idx + 1) if idx < 4 else (idx - 3)}')
plt.show()
"""