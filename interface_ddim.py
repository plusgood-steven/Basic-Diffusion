import argparse
import torch
import torch.nn as nn
import os
import logging
import math
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from dataset import MNIST
from model import UNet, GaussianDiffusion, DDIM
from utils import same_seed,count_parameters,show_result, set_num_workers
#from pytorch_gan_metrics import get_fid

def sample_data(model, batch_size, sample_num, ddim_timesteps, dir_path, device):
    print("Sampling data...")
    model.eval()
    save_dir = os.path.join(dir_path, "output")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    image_id = 0
    while sample_num:
        one_sample_batch = batch_size if sample_num // batch_size else sample_num
        images = model.sample(one_sample_batch, ddim_timesteps=ddim_timesteps, device=device)
        sample_num -= one_sample_batch
        for i in range(len(images)):
            image_id += 1
            save_image(images[i], os.path.join(save_dir , str(image_id).zfill(5) + '.png'))
        print('\r' + 'Sampling data...' + str(image_id), end='')


def sample_grid(model, sample_num, ddim_timesteps, dir_path, device):
    print("Sampling Gird data...")
    model.eval()
    save_dir = os.path.join(dir_path, "grid_output")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    images = model.sample_grid(sample_num, 8, device, ddim_timesteps=ddim_timesteps)
    images = make_grid(images)
    save_image(images, os.path.join(save_dir , 'result.png'))
    print("finished !")
    return


def generate_linear_schedule(T, beta_1, beta_T):
    return torch.linspace(beta_1, beta_T, T).double()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_data_num",default=10000,type=int)
    parser.add_argument("--sample_size",default=2000,type=int)
    parser.add_argument("--ddim_timesteps",default=50,type=int)
    parser.add_argument("--dir_path",default="./ddim_gen",type=str)
    parser.add_argument("--gpu_num",default=0,type=int)
    parser.add_argument("--model_path", default="./weight/best_model.pt")
    parser.add_argument("--only_sample_grid",default=False, type=bool)
 
    params = parser.parse_args()
    device = f"cuda:{params.gpu_num}" if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(params.model_path, map_location=device)
    print("device :",device)

    set_num_workers(cpu_num=8)

    args = checkpoint['args']
    args.sample_data_num = params.sample_data_num
    args.sample_size = params.sample_size
    args.dir_path = params.dir_path
    args.ddim_timesteps = params.ddim_timesteps


    if not os.path.isdir(args.dir_path):
        os.makedirs(args.dir_path)

    info_records = {}

    beta = generate_linear_schedule(args.num_timesteps, args.beta_1, args.beta_T)
    model_UNet = UNet(
        channel_mults=args.channel_mults, 
        base_channels=args.base_channels,
        time_dim=args.time_dim).to(device)
    defussionModel = DDIM(model_UNet, image_size=(28,28), img_channels=3, betas=beta).to(device)
    defussionModel.load_state_dict(checkpoint['model'])

    model_parameters = count_parameters(defussionModel)

    print(f'model parameters: {model_parameters}')
    if not params.only_sample_grid:
        sample_data(defussionModel, args.sample_size, args.sample_data_num, args.ddim_timesteps, args.dir_path, device)
    sample_grid(defussionModel, 8, args.ddim_timesteps ,args.dir_path, device)