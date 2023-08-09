import torch
import numpy as np
from scipy import linalg
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class SingleCholesterolDataset(Dataset):
    def __init__(self, root = './data/hb_hbo2_fat_29', wavelist = np.arange(700, 981, 10), depth = [35], transform = None, whiten = False, normalize = True):
        self.root = root
        self.depth = depth
        self.wavelist = wavelist
        self.transform = transform
        self.whiten = whiten
    
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
            if self.normalize:
                simdata = self.normalize(simdata)
            simdata = simdata.transpose((1, 2, 0)).reshape((h*w, c))
            if self.whiten:
                simdata = simdata.T
                sim_mean = simdata.mean(axis = -1)
                simdata -= sim_mean[:, np.newaxis]
                U, D = linalg.svd(simdata, full_matrices = False, check_finite = False)[:2]
                U *= np.sign(U[0])
                K = (U / D).T
                simdata = np.dot(K, simdata)
                simdata *= np.sqrt(h * w)
            simdata = simdata.T
            simdatalist.append(simdata)
        simdatalist = np.array(simdatalist).transpose(1, 0, 2).reshape((len(self.wavelist), -1))
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