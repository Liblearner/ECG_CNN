import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y

    def __getitem__(self, index):
        x=torch.tensor(self.x[index],dtype=torch.float32)
        y=torch.tensor(self.y[index],dtype=torch.float32)
        return x,y

    def __len__(self):
        return len(self.x)