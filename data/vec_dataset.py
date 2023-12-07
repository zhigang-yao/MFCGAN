import os
import torch
import pandas as pd
import random
from data.base_dataset import BaseDataset


class VecDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host random vectors from domain A '/path/to/data/trainA.csv'
    and random vectors from domain B '/path/to/data/trainB.csv' respectively.
    
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A.csv')  # create a path '/path/to/data/trainA.csv'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B.csv')  # create a path '/path/to/data/trainB.csv'

        self.A = self._load_data_from_excel(self.dir_A)
        self.B = self._load_data_from_excel(self.dir_B)

        self.A_size = len(self.A)  # get the size of dataset A
        self.B_size = len(self.B)  # get the size of dataset B
        self.data_shape = self._get_vector_size()

    def _load_data_from_excel(self, excel_file):
        df = pd.read_csv(excel_file, header=None)
        data = df.values.tolist()
        data = torch.tensor(data)
        return data

    def _get_vector_size(self):
        return (list(self.A[0].shape), list(self.B[0].shape))
    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
        """
        index_A = index % self.A_size  # make sure index is within the range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        
        A = self.A[index_A]
        B = self.B[index_B]

        return {'A': A, 'B': B}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
