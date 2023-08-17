import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from scipy.io import savemat, loadmat

if __name__ == "__main__":
    hbhbo2fat = np.load('./data/hbo2hbchpr_11.npy')
    plot_weights(hbhbo2fat, legend = ["HbO2", "Hb", "Cholesterol"], save = False, final = 951, scale = False, div = 25)

    df1 = pd.read_csv('./data/cholesterol_abs.csv')
    plt.figure(figsize = (10, 6))
    plt.plot(df1['Wavelength'], df1['AbsorptionCoeff'])
    plt.xticks(np.arange(430, 1090, 40))
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (m^-1)")
    plt.show()

    df2 = pd.read_csv('./data/hbo2hb_mec.csv')

    x, y, xp = np.arange(700, 951, 25), np.array([0.6, 0.5, 0.7, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9])/1000, np.arange(700, 981, 10)
    model = np.poly1d(np.polyfit(x, y, 3))
    yp = model(xp)

    hbo2hbchpr = np.zeros((29, 4))
    for idx, wave in enumerate(np.arange(700, 981, 10)):
        hbo2hbchpr[idx, 2] = df1['AbsorptionCoeff'][df1['Wavelength'] == wave].values[0] / 1000
        hbo2hbchpr[idx, 1] = df2['Hb'][df2['Wavelength'] == wave].values[0] * 2.303 * 6.45 /64500
        hbo2hbchpr[idx, 0] = df2['HbO2'][df2['Wavelength'] == wave].values[0] * 2.303 * 6.45 /64500
        hbo2hbchpr[idx, 3] = yp[idx]

    plot_weights(hbo2hbchpr[:,:3], legend = ["HbO2", "Hb", "Cholesterol"], save = False, scale = False, div = 10, final = 981)

    df3 = pd.read_csv('./data/water.csv')
    wx = [df3['lambda'][df3['lambda'] == i].item() for i in range(700, 981, 10) if i in df3['lambda'].values]
    wy = [df3['absorption'][df3['lambda'] == i].item() for i in wx]
    wxp = np.arange(700, 981, 10)
    wmodel = np.poly1d(np.polyfit(wx, wy, 6))
    wyp = wmodel(wxp)

    hbo2hbchwtr = np.zeros((29, 4))
    for idx, wave in enumerate(np.arange(700, 981, 10)):
        hbo2hbchwtr[idx, 2] = df1['AbsorptionCoeff'][df1['Wavelength'] == wave].values[0] / 100
        hbo2hbchwtr[idx, 1] = df2['Hb'][df2['Wavelength'] == wave].values[0] * 2.303 * 6.45 /64500
        hbo2hbchwtr[idx, 0] = df2['HbO2'][df2['Wavelength'] == wave].values[0] * 2.303 * 6.45 /64500
        hbo2hbchwtr[idx, 3] = wyp[idx]
    hbo2hbchwtr[0, 3] = 0.006012

    plot_weights(hbo2hbchwtr[:,:4], legend = ["HbO2", "Hb", "Cholesterol", "Water"], save = False, scale = False, div = 10, final = 981)

    np.save('./data/hbo2hbchwtr_29.npy', hbo2hbchwtr)
    # savemat('./data/hbo2hbchpr.mat', {'hbo2hbchpr': hbo2hbchpr})

    plt.figure(figsize = (10, 6))
    plt.plot(np.arange(700, 981, 10), hbo2hbchpr)
    plt.xticks(np.arange(700, 981, 20))
    plt.legend(['HbO2', 'Hb', 'Cholesterol', 'Prostate'])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.show()

    plt.figure(figsize = (10, 6))
    plt.scatter(x, y)
    plt.plot(np.arange(700, 981, 10), hbo2hbchpr[:, 3], 'r')
    plt.xticks(np.arange(700, 981, 20))
    plt.legend(['Prostate Ground Truth', 'Model Prediction'])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.ylim(0, 0.0012)
    plt.show()

    plt.figure(figsize = (10, 6))
    plt.scatter(wx, wy)
    plt.plot(wxp, wyp, 'r')
    plt.xticks(np.arange(700, 981, 20))
    plt.legend(['Water Ground Truth', 'Model Prediction'])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.show()

    """
    arr = np.load('./data/hbo2hbchpr_57.npy')
    plt.figure(figsize = (10, 6))
    plt.plot(np.arange(700, 981, 10), hbo2hbchpr)
    plt.xticks(np.arange(700, 981, 20))
    plt.legend(['HbO2', 'Hb', 'Cholesterol', 'Prostate'])
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorption Coefficient (cm^-1)")
    plt.show()

    for i in range(hbo2hbchwtr.shape[0]):
        print(f"{hbo2hbchwtr[i, 0]:.5f} {hbo2hbchwtr[i, 1]:.5f} {hbo2hbchwtr[i, 2]:.5f} {hbo2hbchwtr[i, 3]:.5f}")
    """

"""
    HbO2, Hb, Cholesterol, Melanin(Skin)
    data = np.load('./data/hbo2hbchpr_57.npy')
    for idx, wave in enumerate(list(range(700, 981, 5))):
        print(f'{wave} {data[idx][0]:.5f} {data[idx][1]:.5f} {data[idx][2]:.5f} {data[idx][3]:.5f} {1.70 * 10e12 * ((wave)**(-3.48)):.5f}')   
    HbO2, Hb, Cholesterol, Carbon Tissue Layer
    for idx, wave in enumerate(list(range(700, 981, 5))):
        print(f'{wave} {data[idx][0]:.5f} {data[idx][1]:.5f} {data[idx][2]:.5f} {data[idx][3]:.5f} {(12 / 2 / 25 * 10e4 * math.exp(-0.0032 * wave)):.5f}')
"""