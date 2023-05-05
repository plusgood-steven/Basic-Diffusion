import argparse
import torch
import torch.nn as nn
import os
import logging
import math
from torchvision.utils import save_image
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from dataset import MNIST
from model import UNet, GaussianDiffusion
from utils import same_seed, count_parameters, show_result, set_num_workers
from pytorch_gan_metrics import get_fid


def train(train_loader, model, args, device):
    n_epochs, best_accu, best_loss, step, early_stop_count = args.n_epochs, 0, math.inf, 0, 0
    scheduler = None
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # if config['scheduler'] == "cosine":
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    # if config['scheduler'] == "reduce":
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.7, patience=5)

    logging.info(
        f"Optimizer:{optimizer.__class__.__name__}")
    logging.info(f"config:{args}")

    train_loss_records = []
    Best_FID = FID = 100000000

    print("start train")

    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x in train_pbar:
            optimizer.zero_grad()
            x = x.to(device)
            loss = model(x)
            loss.backward()
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(
                f'Epoch [{epoch + 1}/{n_epochs}] training')
            train_pbar.set_postfix(
                {'loss': loss.detach().item(), 'lr': optimizer.param_groups[0]['lr']})

        mean_train_loss = sum(loss_record)/len(loss_record)
        train_loss_records.append(mean_train_loss)

        # 計算
        if (epoch + 1) % args.sample_num == 0:
            FID = sample_data(model, args.sample_size, args.sample_data_num,
                              dir_path=args.dir_path, fid_path=args.fid_path, device=device)

        if scheduler:
            if args.scheduler == "reduce":
                scheduler.step(mean_train_loss)
            else:
                scheduler.step()

        if FID < Best_FID:
            Best_FID = FID
            # Save best accuracy model
            torch.save({
                "model": model.state_dict(),
                "epochs": epoch,
                "args": args}, f"{args.dir_path}/best_model.pt")
            print(
                'Saving model with best FID {:.3f}...'.format(Best_FID))
            logging.info(
                'Saving model with best FID {:.3f}...'.format(Best_FID))

        print(
            f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f} Best FID: {Best_FID:.4f} FID: {FID:.4f}')
        logging.info(
            f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f} Best FID: {Best_FID:.4f} FID: {FID:.4f}')

        torch.save({
            "model": model.state_dict(),
            "epochs": epoch,
            "optimizer": optimizer.state_dict(),
            "args": args}, f"{args.dir_path}/last_model.pt")

    return train_loss_records, None


def sample_data(model, batch_size, sample_num, dir_path, fid_path, device):
    print("Sampling data...")
    model.eval()
    save_dir = os.path.join(dir_path, "output")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    image_id = 0
    all_images = None
    while sample_num > 0:
        one_sample_batch = batch_size if sample_num // batch_size else sample_num
        images = model.sample(one_sample_batch, device)
        all_images = torch.cat((all_images, images),
                               dim=0) if all_images else images
        sample_num -= one_sample_batch
        for i in range(len(images)):
            image_id += 1
            save_image(images[i], os.path.join(
                save_dir, str(image_id).zfill(5) + '.png'))
        print('\r' + 'Sampling data...' + str(image_id), end='')
    all_images = all_images.view(-1, 3, 28, 28)
    all_images = all_images.mul(255).add_(0.5).clamp_(
        0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
    to_tensor = transforms.ToTensor()
    all_images = torch.stack([to_tensor(image) for image in all_images], dim=0)
    print("\n")
    return get_fid(all_images, fid_path)


def generate_linear_schedule(T, beta_1, beta_T):
    return torch.linspace(beta_1, beta_T, T).double()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", default=200, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--beta_1", default=1e-4, type=float)
    parser.add_argument("--beta_T", default=1e-2, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--num_timesteps", default=1000, type=int)
    parser.add_argument("--dir_path", default="./results", type=str)
    parser.add_argument("--gpu_num", default=0, type=int)
    parser.add_argument("--dataset_path", default="./data", type=str)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--scheduler", default=None, type=str)
    parser.add_argument("--sample_num", default=20, type=int)
    parser.add_argument("--base_channels", default=64, type=int)
    parser.add_argument("--time_dim", default=128, type=int)
    parser.add_argument("--channel_mults",
                        default=(1, 2, 2), type=int, nargs='+')
    parser.add_argument("--fid_path", default="./weight/mnist.npz", type=str)
    parser.add_argument("--sample_data_num", default=10000, type=int)
    parser.add_argument("--sample_size", default=1000, type=int)

    args = parser.parse_args()

    device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu'
    print("device :", device)
    same_seed(123456)
    set_num_workers(cpu_num=8)

    train_dataset = MNIST()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, num_workers=8)

    if not os.path.isdir(args.dir_path):
        os.makedirs(args.dir_path)

    info_records = {}

    logging.basicConfig(
        filename=args.dir_path + '/train.log', format='%(levelname)s:%(message)s', encoding='utf-8', level=logging.INFO)

    beta = generate_linear_schedule(
        args.num_timesteps, args.beta_1, args.beta_T)
    model_UNet = UNet(
        channel_mults=args.channel_mults,
        base_channels=args.base_channels,
        time_dim=args.time_dim).to(device)
    diffusionModel = GaussianDiffusion(model_UNet, image_size=(
        28, 28), img_channels=3, betas=beta).to(device)

    model_parameters = count_parameters(diffusionModel)
    logging.info(f'args: {args}')
    logging.info(f'model parameters: {model_parameters}')
    logging.info(f'model: {diffusionModel}')
    print(f'model parameters: {model_parameters}')

    train(train_loader, diffusionModel, args, device)

    # info_records = {"train_loss": train_loss_records, "val_accu": val_accu_records}

    # show_result(info_records, config["dir_path"])
