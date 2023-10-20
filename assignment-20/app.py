import os
import numpy
import torch
from torch import autocast
from torchvision import transforms as tfms

import PIL
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel, KDPM2DiscreteScheduler

# For video display:
from IPython.display import HTML
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
madhubani = torch.load("./madhubani-art/learned_embeds.bin")
kuvishnov = torch.load("./kuvshinov/learned_embeds.bin")
line_art_embed = torch.load("./line-art/learned_embeds.bin")
walter_wick_embed = torch.load("./walter-wick-photography/learned_embeds.bin")
birb_embed = torch.load('./birb-style/learned_embeds.bin')
indian_water_color = torch.load("./indian-watercolor-portraits/learned_embeds.bin")
herge_embed = torch.load("./herge-style/learned_embeds.bin")
lucky_luke = torch.load("./lucky-luke/learned_embeds.bin")


pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16).to(device)



def blue_loss(images):
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel
    return error

def qr_loss(images, qr_img):
    error = torch.abs(images - qr_img).mean()    
    return error

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)


def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images