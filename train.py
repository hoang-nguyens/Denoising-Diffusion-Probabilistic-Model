import copy
from tqdm.notebook import  tqdm
from utils import get_data, EMA, plot_images, save_images
from modules import Unet, Diffusion
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
def train(args):
    device = args.device
    dataloader = get_data(args)
    model = Unet(num_class=args.num_class).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(image_size=args.image_size, device=args.device)
    logger = SummaryWriter(os.path.join('runs', args.run_name))
    l = len(dataloader)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            t = diffusion.sample_timestep(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_image(images, t)
            if np.random.random() < 0.1:
                labels = None
            predicted_nosie = model(x_t, t, labels)
            mse_loss = mse(predicted_nosie, noise)
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            pbar.set_postfix(MSE=mse_loss.item())

            if epoch % 10 == 0:
                labels = torch.arange(10).long().to(device)
                sampled_images = diffusion.sample(model, n=len(labels), labels=labels)
                ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
                plot_images(sampled_images)
                save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
                save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
                torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
                torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
                torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs  = 300
    args.batch_size = 12
    args.image_size = 64
    args.num_class = 10
    args.dataset_path = r""
    args.device = 'cuda'
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    launch()








