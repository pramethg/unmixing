import torch
import numpy as np
from scipy.io import loadmat
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class SingleCholesterolDataset(Dataset):
    def __init__(self, root = './data/hb_hbo2_fat_29', wavelist = np.arange(700, 981, 10), depth = 35, transform = None):
        self.root = root
        self.depth = depth
        self.wavelist = wavelist
        self.transform = transform
    
    def __len__(self):
        return 1
    
    def __repr__(self):
        return f"Cholesterol Dataset(Root: {self.root}_{self.depth}/, Wavelengths: {self.__len__()}, Depth: {self.depth})"
    
    def __getitem__(self, idx):
        simdata = np.array([np.array(loadmat(f'{self.root}_{self.depth}/PA_Image_{wave}.mat')['Image_PA']) for wave in self.wavelist])
        c, h, w = simdata.shape
        simdata = simdata.transpose((1, 2, 0)).reshape((h*w, c))
        if self.transform:
            simdata = self.transform(simdata)
        for wave in range(len(self.wavelist)):
            simdata[:,wave] -= np.min(simdata[:,wave])
            simdata[:,wave] /= np.max(simdata[:,wave])
        return torch.Tensor(simdata)

class MultipleCholesterolDataset:
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
        if self.transform:
            sim_data = self.transform(depth_data)
        return torch.Tensor(depth_data)

if __name__ == "__main__":
    data = SingleCholesterolDataset()
    dataloader = DataLoader(data, 1, True)
    print(data[0].shape)