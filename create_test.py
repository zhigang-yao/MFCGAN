"""
This script is used to create a saved models and dataset for the trained model.
The saved models are saved in the directory opt.res_dir/opt.name/
The dataset is saved in the directory opt.res_dir/opt.name/dataset.pth
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

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers


    load_suffix = '%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
    model.load_networks(load_suffix)
    model.eval()

    # create a directory under opt.res_dir to save the models
    if not os.path.exists(opt.res_dir):
        os.makedirs(opt.res_dir)
    save_dir = os.path.join(opt.res_dir, opt.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # use torch to save the model
    torch.save(model.netG_A, os.path.join(save_dir, 'netG_A.pth'))
    torch.save(model.netG_B, os.path.join(save_dir, 'netG_B.pth'))

    # save the dataset to the save_dir
    torch.save(dataset, os.path.join(save_dir, 'dataset.pth'))

