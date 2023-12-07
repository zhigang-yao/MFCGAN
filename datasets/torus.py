import numpy as np
from torch.utils.data import Dataset
import os


class TORUS(Dataset):
    def __init__(self, **kwargs):
        class_directory = os.path.dirname(os.path.abspath(__file__))
        data_directory = os.path.join(class_directory, 'torus', 'B.csv')
        self.x = np.genfromtxt(data_directory, delimiter=',',dtype=np.float32)
        self.input_size = self.x.shape[1]
        self.y = self.x*0 // 1
        self.label_size = 1
        self.dataset_size = self.x.shape[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        return self.x[i], self.y[i]