# Manifold Fitting with CycleGAN

## Pre-requisites
- Python 3.7+
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
Examples can be found in the notebooks, which is self-contained and can be run independently.

### Installation

- Clone this repo:
```bash
git clone https://https://github.com/zhigang-yao/MFCGAN
cd MFCGAN
```

- Install [PyTorch](http://pytorch.org) and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  You can do this by running the following:
```bash
pip install -r requirements.txt
```

### MFCGAN train/test
- The data sets are included in the `./datasets` folder.
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
```bash
python train.py --dataroot ./datasets/circle --name circle
```
- Test the model, create the sub-networks and save the results:
```bash
python python create_test.py --dataroot ./datasets/circle --name circle
```
- The generators and data set will be save as *.pth* file in the folder `./results/circle/`.

### MFCGAN with images
- The data sets are included in the `./datasets` folder.
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
```bash
python train.py --dataroot ./datasets/gif/ --name gif --no_flip  --model vec2pic --dataset_mode vec2pic
```
- Test the model, create the sub-networks and save the results:
```bash
python create_test.py --dataroot ./datasets/gif/ --name gif_test --no_flip --model vec2pic --dataset_mode vec2pic --gpu_ids -1
```
- The generators and data set will be save as *.pth* file in the folder `./results/circle/`.


## Citation
If you use this code for your research, please cite our papers.
```
@article{yao2023manifold,
  title={Manifold Fitting with CycleGAN},
  author={Yao, Zhigang and Su, Jiaji and Yau, Shing-Tung},
  journal={Technical Report},
  year={2023}
}
```