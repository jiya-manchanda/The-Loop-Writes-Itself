import os
from pathlib import Path
from PIL import Image
import numpy as np

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

def save_mask(mask_arr, path):
    """
    mask_arr: HxW numpy array, values 0/255 or 0/1
    """
    img = Image.fromarray((mask_arr > 0).astype('uint8') * 255)
    img.save(path)

def save_rgb(rgb_arr, path):
    """
    rgb_arr: HxWx3 numpy array, 0-255
    """
    Image.fromarray(rgb_arr.astype('uint8')).save(path)

def load_mask(path):
    im = Image.open(path).convert("L")
    return np.asarray(im)

def load_rgb(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im)
