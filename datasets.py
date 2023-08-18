import torch
import numpy as np
from scipy import linalg
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

class ZCA(BaseEstimator, TransformerMixin):
    def __init__(self, regularization=1e-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = as_float_array(X, copy=self.copy)
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_
        sigma = np.dot(X.T, X) / (X.shape[0] - 1)
        U, S, V = np.linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1 / np.sqrt(S + self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed

class SingleCholesterolDataset(Dataset):
    def __init__(self, root = './data/hb_hbo2_fat_11', wavelist = 'EXP10', depth = [35], transform = None, whiten = 'zca', normalize = True):
        self.root = root
        self.depth = depth
        self.wavelist = wavelist
        self.transform = transform
        self.whiten = whiten
        wavelists = {'FULL': np.arange(700, 981, 10), 'EXP10': [750, 760, 800, 850, 900, 910, 920, 930, 940, 950], 'EXP6': [750, 760, 800, 850, 900, 925]}
        self.wavelist = wavelists[wavelist]
    
    @staticmethod
    def normalize(image):
        imgmin = np.min(image, axis = (1, 2), keepdims = True)
        imgmax = np.max(image, axis = (1, 2), keepdims = True)
        image = (image - imgmin) / (imgmax - imgmin)
        return image
    
    def __len__(self):
        return len(self.depth)
    
    def __repr__(self):
        return f"Cholesterol Dataset(Root: {self.root}_{self.depth}/, Wavelengths: {len(self.wavelist)}, Depth: {self.depth})"
    
    def __getitem__(self, idx):
        simdatalist = []
        for depth in self.depth:
            simdata = np.array([np.array(loadmat(f'{self.root}_{depth}/PA_Image_{wave}.mat')['Image_PA']) for wave in self.wavelist])
            c, h, w = simdata.shape
            simdata = self.normalize(simdata).transpose((1, 2, 0)).reshape((h*w, c))
            if self.whiten == 'zca':
                zca = ZCA()
                simdata = zca.fit_transform(simdata)
            if self.whiten == 'ica':
                simdata = simdata.T
                sim_mean = simdata.mean(axis = -1)
                simdata -= sim_mean[:, np.newaxis]
                U, D = linalg.svd(simdata, full_matrices = False, check_finite = False)[:2]
                U *= np.sign(U[0])
                K = (U / D).T
                simdata = np.dot(K, simdata)
                simdata *= np.sqrt(h * w)
            if self.normalize:
                simdata = simdata.reshape((h, w, c)).transpose((2, 0, 1))
                simdata = self.normalize(simdata).transpose((1, 2, 0)).reshape((h*w, c))
            simdatalist.append(simdata)
        simdatalist = np.array(simdatalist).transpose(2, 0, 1).reshape((len(self.wavelist), -1)).T
        if self.transform:
            simdata = self.transform(simdata)
        return torch.Tensor(simdatalist)

class MultipleCholesterolDataset(Dataset):
    def __init__(self, root = "./data", wavelist = np.arange(700, 981, 10), depths = np.arange(15, 41, 5), transform = None):
        self.root = root
        self.depths = depths
        self.wavelist = wavelist
        self.transform = transform

    def __len__(self):
        return len(self.depths)
    
    def __repr__(self):
        return f"Cholesterol Dataset(Root: {self.root}, Wavelengths: {len(self.wavelist)}, Depths: {len(self.depths)}"
    
    def __getitem__(self, idx):
        depth = self.depths[idx]
        depth_data = np.array([np.array(loadmat(f'{self.root}/hb_hbo2_fat_29_{depth}/PA_Image_{wave}.mat')['Image_PA']) for wave in self.wavelist])
        for wave in range(len(self.wavelist)):
            depth_data[wave] -= np.min(depth_data[wave])
            depth_data[wave] /= np.max(depth_data[wave])
        if self.transform:
            sim_data = self.transform(depth_data)
        return torch.Tensor(depth_data)

def test():
    dataset = SingleCholesterolDataset(wavelist = np.arange(700, 981, 10), depth = list(range(15, 41, 5)), normalize = True)
    print(dataset[0].shape)

if __name__ == "__main__":
    test()