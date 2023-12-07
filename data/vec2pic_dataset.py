import os
import pandas as pd
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class Vec2PicDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host random vectors from domain A '/path/to/data/A.csv'
    and images from domain B '/path/to/data/B' respectively.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A.csv')  # create a path '/path/to/data/A.csv'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/B'

        self.A = self._load_data_from_excel(self.dir_A)
        self.A_size = len(self.A)  # get the size of dataset A

        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/B'
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.data_shape = self._get_vector_size()

    def _load_data_from_excel(self, excel_file):
        df = pd.read_csv(excel_file, header=None)
        data = df.values.tolist()
        data = torch.tensor(data)
        return data

    def _get_vector_size(self):
        return (list(self.A[0].shape),list(self.__getitem__(0)['B'].shape)) # (vector_size, image_size)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            B_paths (str)    -- image paths
        """
        index_A = index % self.A_size  # make sure index is within the range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        B_img = Image.open(B_path).convert('RGB')
        # apply image transformation
        A = self.A[index_A]
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
