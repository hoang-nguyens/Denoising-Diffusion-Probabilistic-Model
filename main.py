from modules import Diffusion, Unet
from utils import *
import torch
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 5
args.dataset_path = r""
args.labels = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diffusion = Diffusion
model = Unet()
checkpoint = torch.load('ckpt.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

dataloader = get_data(args)
test_batch = next(iter(dataloader))[0]

images = diffusion.sample(model, args.batch_size, args.labels)

plot_images(images)