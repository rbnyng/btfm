# calculates mean and standard deviation over whole california dataset
# (by band)
from data import SentinelTimeSeriesDataset
import numpy as np
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import os

base_path = "./"
datasets = ["california", "wyoming"]

# set up logging. Include time and level name
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

with logging_redirect_tqdm():
    for dataset_name in datasets:
        tiles = os.listdir(base_path + f'data/{dataset_name}/processed')

        # keep all tiles that start with MGRS
        tiles = [tile for tile in tiles if tile.startswith('MGRS')]

        num_tiles = len(tiles)

        logging.info(f"{dataset_name}: tiles: {num_tiles}")

        # 11 bands
        band_means = np.zeros(11, dtype=np.float64)
        band_stds = np.zeros(11, dtype=np.float64)
        total_samples = 0

        for tile in tqdm(tiles):
            logging.info(f"{dataset_name}: tile {tile}")
            bands = np.load(base_path + f'data/{dataset_name}/processed/{tile}/bands.npy')
            masks = np.load(base_path + f'data/{dataset_name}/processed/{tile}/masks.npy')
            # bands is t, h, w, bands
            # masks is t, h, w

            bands = np.clip(bands.astype(np.float32)/10000, 0, 1)

            # only use bands where we have valid data (mask = 1)
            bands = bands[masks == 1]

            # calculate mean and std for each band
            band_means += bands.mean(axis=0)
            band_stds += bands.std(axis=0)

        # print band_means and band_stds
        band_means /= num_tiles
        band_stds /= num_tiles
        logging.info(f"Band means: {band_means}")
        logging.info(f"Band stds: {band_stds}")
