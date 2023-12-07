"""
This script is used to create a test dataset for the trained model.
The test dataset is saved in the directory opt.res_dir/opt.name/testset.pth
We use this script to keep all the opts are the same as the training opts.
"""
from options.train_options import TrainOptions
import os
from data import create_dataset
from models import create_model
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    opt.isTrain = False             # set to False to avoid creating useless models

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    opt.input_dim, opt.output_dim = dataset.dataset.data_shape # get the size of the input/output vector
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)


    # create a directory under opt.res_dir to save the models
    if not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir)
    save_dir = os.path.join(opt.res_dir, opt.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save the dataset to the save_dir
    torch.save(dataset, os.path.join(save_dir, 'testset.pth'))

