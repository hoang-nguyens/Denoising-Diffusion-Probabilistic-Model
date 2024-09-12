import os
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def plot_images(images, figsize=(32, 32)):
    # images shape: [batches, channels, height, width]
    batches = images.shape[0]
    fig, axes = plt.subplots(1, batches, figsize=figsize)

    for i in range(batches):
        img = images[i].numpy().transpose(1, 2, 0)
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(args):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),
        torchvision.transforms.RandomResizedCrop(args.image_size, scale = (0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(args.dataset_path, transform = transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle = True)
    return dataloader

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0
    def updata_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(ma_model.parameters(), current_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        else:
            return old*self.beta + (1 - self.beta)*new
    def step_ema(self, ema_model, model, step_start_ema = 2000):
        if self.step < self.step_ema:
            self.reset_parameters(ema_model, model)
            self.step +=1
            return
        self.updata_model_average(ema_model, model)
        self.step +=1
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
