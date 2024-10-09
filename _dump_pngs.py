import torch
import os
import numpy as np
import random
import logging
from PIL import Image

logging.basicConfig(level=logging.INFO)

dataset_name = "california"
base_path = "./"
min_valid_pixels = 32

tiles = os.listdir(base_path + f'data/{dataset_name}/processed')

def get_tile_data(tile):
    bands = np.load(base_path + f'data/{dataset_name}/processed/{tile}/bands.npy')
    masks = np.load(base_path + f'data/{dataset_name}/processed/{tile}/masks.npy')
    bands_mean = np.load(base_path + f'data/{dataset_name}/processed/{tile}/band_mean.npy')
    bands_std = np.load(base_path + f'data/{dataset_name}/processed/{tile}/band_std.npy')
    # Convert to tensor and standardize bands
    bands = torch.tensor(bands).float()
    masks = torch.tensor(masks)
    bands_mean = torch.tensor(bands_mean)
    bands_std = torch.tensor(bands_std)
    bands = (bands - bands_mean) / bands_std

    masks_sum = masks.sum(dim=0)
    gt_min_valid = masks_sum > min_valid_pixels
    fast_valid_indices = np.nonzero(gt_min_valid)

    return bands, masks, fast_valid_indices

# for each tile in the dataset we will create a tmp/{tile}/{time} directory
# in that directory will be a png image for each band
# the bands will be greyscale with red being used for any pixels that are
# masked out

for tile in tiles:
    bands, masks, _ = get_tile_data(tile)
    # bands is t, h, w, bands
    # masks is t, h, w

    # create a tmp directory
    os.makedirs("tmp", exist_ok=True)
    os.makedirs(f"tmp/{tile}", exist_ok=True)

    """for t in range(bands.shape[0]):
        logging.info(f"Processing tile {tile} time {t} of {bands.shape[0]}")
        for b in range(bands.shape[3]):
            logging.info(f"Processing band {b} of {bands.shape[3]}")
            band = bands[t, :, :, b]
            mask = masks[t, :, :]
            band = band - band.min()
            band = band / band.max()
            band = (band * 255).numpy().astype(np.uint8)
            mask = mask.numpy().astype(np.uint8)
            mask = 255 - (mask * 255)
            mask = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
            band = np.stack([band, band, band], axis=-1)
            band[mask > 0] = mask[mask > 0]
            band = Image.fromarray(band)
            band.save(f"tmp/{tile}/{t}_{b}.png")"""

    # now we want to create a composite of the tile. For each band, w and h we use all
    # pixels with mask = 1 to create the composite

    logging.info(f"Creating composite for tile {tile}")

    composite_sum = np.zeros((bands.shape[1], bands.shape[2], bands.shape[3]), dtype=np.float32)
    composite_count = np.zeros((bands.shape[1], bands.shape[2], bands.shape[3]), dtype=np.float32)

    for t in range(bands.shape[0]):
        for b in range(bands.shape[3]):
            band = bands[t, :, :, b]
            mask = masks[t, :, :]
            band = band.numpy()
            mask = mask.numpy()
            composite_sum[mask > 0, b] += band[mask > 0]
            composite_count[mask > 0, b] += 1

    composite = composite_sum / composite_count

    # now create a composite png for each band
    for b in range(bands.shape[3]):
        logging.info(f"Processing band {b} of {bands.shape[3]}")
        band = composite[:, :, b]
        band = band - band.min()
        band = band / band.max()
        band = (band * 255).astype(np.uint8)
        band = Image.fromarray(band)
        band.save(f"tmp/{tile}/composite_{b}.png")

    # create an RGB composite from the first 3 bands (RBG, will need to convert)
    rgb = composite[:, :, :3]
    rgb = rgb - rgb.min()
    rgb = rgb / rgb.max()
    rgb = rgb[:, :, [0, 2, 1]]
    rgb = (rgb * 255).astype(np.uint8)
    rgb = Image.fromarray(rgb)
    rgb.save(f"tmp/{tile}/composite_rgb.png")
