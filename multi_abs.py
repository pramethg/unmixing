import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from scipy.io import savemat, loadmat

if __name__ == "__main__":

    df1 = pd.read_csv('./data/cholesterol_abs.csv')
    df2 = pd.read_csv('./data/hbo2hb_mec.csv')

    x, y, xp = np.arange(700, 951, 25), np.array([0.6, 0.5, 0.7, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9])/1000, np.arange(700, 981, 5)
    model = np.poly1d(np.polyfit(x, y, 5))
    yp = model(xp)

    hbo2hbchpr = np.zeros((57, 4))
    for idx, wave in enumerate(np.arange(700, 981, 5)):
        hbo2hbchpr[idx, 2] = df1['AbsorptionCoeff'][df1['Wavelength'] == wave].values[0] / 1000
        if wave not in df2['Wavelength'].values:
            hbo2hbchpr[idx, 1] = ((df2['Hb'][df2['Wavelength'] == (wave - 1)].values[0]) + (df2['Hb'][df2['Wavelength'] == (wave + 1)].values[0])) * 2.303 * 6.45 /64500 /2
            hbo2hbchpr[idx, 0] = ((df2['HbO2'][df2['Wavelength'] == (wave - 1)].values[0]) + (df2['HbO2'][df2['Wavelength'] == (wave + 1)].values[0])) * 2.303 * 6.45 /64500 /2
        else:
            hbo2hbchpr[idx, 1] = df2['Hb'][df2['Wavelength'] == wave].values[0] * 2.303 * 6.45 /64500
            hbo2hbchpr[idx, 0] = df2['HbO2'][df2['Wavelength'] == wave].values[0] * 2.303 * 6.45 /64500
        hbo2hbchpr[idx, 3] = yp[idx]

    # plot_weights(hbo2hbchpr[:,:3], legend = ["HbO2", "Hb", "Cholesterol"], save = False, scale = False, div = 5, final = 981)
    # np.save('./data/hbo2hbchpr_57.npy', hbo2hbchpr)
    # savemat('./data/hbo2hbchpr.mat', {'hbo2hbchpr': hbo2hbchpr})

    # """
    plt.figure(figsize = (10, 6))
    plt.plot(np.arange(700, 981, 5), hbo2hbchpr)
    plt.xticks(np.arange(700, 981, 20))
    plt.legend(['HbO2', 'Hb', 'Cholesterol', 'Prostate'])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (mm^-1)")
    plt.show()
    """
    plt.figure(figsize = (10, 6))
    plt.scatter(x, y)
    plt.plot(np.arange(700, 981, 5), hbo2hbchpr[:, 3], 'r')
    plt.xticks(np.arange(700, 981, 20))
    plt.legend(['Prostate Ground Truth', 'Model Prediction'])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.ylim(0, 0.0012)
    plt.show()
    """

    plt.figure(figsize = (10, 6))
    plt.plot(df1['Wavelength'], df1['AbsorptionCoeff'] / 100)
    plt.plot(df2['Wavelength'], df2['HbO2']*2.303*6.45/64500)
    plt.plot(df2['Wavelength'], df2['Hb']*2.303*6.45/64500)
    plt.xticks(np.arange(430, 1090, 40))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.legend(['Cholesterol', 'HbO2', 'Hb'])
    plt.show()