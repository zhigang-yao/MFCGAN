import os
import torch
# define a function to load the model and dataset from dir
def load_test(dir):
    G_A = torch.load(os.path.join(dir, 'netG_A.pth'))
    G_B = torch.load(os.path.join(dir, 'netG_B.pth'))
    dataset = torch.load(os.path.join(dir, 'dataset.pth'))
    return G_A, G_B, dataset