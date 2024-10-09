# dataset for sentinel 2 data
# in the data/{dataset_name} directory is a directory per MGRS tile
# inside that is masks.npy and bands.npy which are (time, width, height) uint8 and (time, width, height, bands) uint16 respectively
import torch
from torch.utils.data import IterableDataset, get_worker_info
import os
import numpy as np
import random

class SentinelTimeSeriesDatasetForDownstreaming(IterableDataset):
    def __init__(self, dataset_type, min_valid_pixels, transform, shuffle=True):
        self.min_valid_pixels = min_valid_pixels
        self.transform = transform
        self.dataset_type = dataset_type
        self.shuffle = shuffle
        self.tiles = os.listdir(base_downstream_path + f'{dataset_type}')
        self.len_cache = None

        # Store all tile data (workers will handle different tiles)
        self.all_bands = []
        self.all_masks = []
        self.all_doys = []
        self.all_labels = []

        # Find max time dimension
        self.max_time = self._find_max_time()

    def __len__(self):
        if self.len_cache is None:
            total = 0
            for tile in self.tiles:
                masks = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/masks.npy')
                masks_sum = masks.sum(axis=0)
                gt_min_valid = masks_sum > self.min_valid_pixels
                total += gt_min_valid.sum().item()

            self.len_cache = total

        return int(self.len_cache)

    def _find_max_time(self):
        """
        Find the maximum time dimension across all tiles.
        """
        max_time = 0
        for tile in self.tiles:
            bands = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/bands.npy')
            max_time = max(max_time, bands.shape[0])
        return max_time

    def _load_tile(self, tile):
        """
        Load and process a single tile, returning bands, masks, doys, and labels.
        """
        bands = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/bands.npy')
        masks = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/masks.npy')
        doys = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/doys.npy')
        labels = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/labels.npy')
        labels = np.digitize(labels, bins=[10, 20, 30, 40, 50, 60, 70, 80, 90]) - 1

        bands_mean = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/band_mean.npy')
        bands_std = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/band_std.npy')
        bands = (bands - bands_mean) / bands_std

        # Resize dimensions to the max time dimension
        bands = self._resize_time_dimension(bands)
        masks = self._resize_time_dimension(masks)
        doys = self._resize_day_of_year(doys)

        return bands, masks, doys, labels

    def _resize_time_dimension(self, array):
        current_time = array.shape[0]
        if current_time < self.max_time:
            pad_width = [(0, self.max_time - current_time)] + [(0, 0)] * (len(array.shape) - 1)
            array = np.pad(array, pad_width, mode='constant', constant_values=0)
        elif current_time > self.max_time:
            array = array[:self.max_time]
        return array

    def _resize_day_of_year(self, doy):
        current_length = doy.shape[0]
        if current_length < self.max_time:
            pad_width = [(0, self.max_time - current_length)]
            doy = np.pad(doy, pad_width, mode='constant', constant_values=0)
        elif current_length > self.max_time:
            doy = doy[:self.max_time]
        return doy

    def _get_worker_tiles(self):
        """
        Partition the tiles across different workers.
        """
        worker_info = get_worker_info()
        if worker_info is None:  # Single-process data loading
            return self.tiles
        
        # Split the tiles across workers
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        per_worker_tiles = np.array_split(self.tiles, num_workers)
        
        return per_worker_tiles[worker_id]

    def __iter__(self):
        def generate():
            worker_tiles = self._get_worker_tiles()

            for tile in worker_tiles:
                bands, masks, doys, labels = self._load_tile(tile)
                total_pixels = bands.shape[1] * bands.shape[2]
                flat_indices = np.arange(total_pixels)

                if self.shuffle:
                    np.random.shuffle(flat_indices)

                for flat_index in flat_indices:
                    i = flat_index // bands.shape[2]
                    j = flat_index % bands.shape[2]

                    mask_sample = masks[:, i, j]
                    if mask_sample.sum() > self.min_valid_pixels:
                        band_sample = bands[:, i, j]
                        label_sample = labels[i, j]
                        doy_sample = doys[:]

                        # Convert to tensor
                        band_sample = torch.tensor(band_sample)
                        mask_sample = torch.tensor(mask_sample)
                        doy_sample = torch.tensor(doy_sample)
                        label_sample = torch.tensor(label_sample)

                        # Add label as padding to band sample
                        padding_tensor = torch.ones(band_sample.shape[0], 1) * label_sample
                        band_sample = torch.cat((band_sample, padding_tensor), dim=1)

                        # Apply transformation
                        sample = self.transform(mask_sample, band_sample, doy_sample)
                        yield sample

        return generate()


# base_path = "./"
# base_downstream_path = "data-downstream"
base_path = "../../sj514/btfm/"
base_downstream_path = "../btfm-data-preparation/"

class SentinelTimeSeriesDataset(IterableDataset):
    def __init__(self, dataset_name, min_valid_pixels, transform, shuffle=True, buffer_size=6_000_00, shuffle_worker_id=None):
        self.min_valid_pixels = min_valid_pixels
        self.transform = transform
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.tiles = os.listdir(base_path + f'data/{dataset_name}/processed')
        # only select folders
        self.tiles = [tile for tile in self.tiles if os.path.isdir(base_path + f'data/{dataset_name}/processed/{tile}')]
        # debug: choose 10 tiles
        # self.tiles = self.tiles[1:11]
        self.total_valid_samples = 0
        self.buffer = []
        self.buffer_size = buffer_size
        self.len_cache = None
        self.shuffle_worker_id = shuffle_worker_id

    def __len__(self):
        if self.len_cache is None:
            total = 0
            for tile in self.tiles:
                masks = np.load(base_path + f'data/{self.dataset_name}/processed/{tile}/masks.npy')
                # masks is t, h, w
                # we want to count the number of valid pixels
                masks_sum = masks.sum(axis=0)
                gt_min_valid = masks_sum > self.min_valid_pixels
                total += gt_min_valid.sum().item()

            self.len_cache = total

        return int(self.len_cache)

    def __iter__(self):
        def generate():
            worker_info = get_worker_info()

            if self.shuffle_worker_id is not None:
                random.seed(worker_info.id)
            else:
                random.seed()
                random.shuffle(self.tiles)

            if worker_info is None:
                worker_tiles = self.tiles
            else:
                worker_tiles = self.tiles[worker_info.id::worker_info.num_workers]

            for tile in worker_tiles:
                bands = np.load(base_path + f'data/{self.dataset_name}/processed/{tile}/bands.npy')
                masks = np.load(base_path + f'data/{self.dataset_name}/processed/{tile}/masks.npy')
                doys = np.load(base_path + f'data/{self.dataset_name}/processed/{tile}/doys.npy')
                bands_mean = np.load(base_path + f'data/{self.dataset_name}/processed/{tile}/band_mean.npy')
                bands_std = np.load(base_path + f'data/{self.dataset_name}/processed/{tile}/band_std.npy')
                # Convert to tensor and standardize bands
                bands = torch.tensor(bands).float()
                masks = torch.tensor(masks)
                doys = torch.tensor(doys)
                bands_mean = torch.tensor(bands_mean)
                bands_std = torch.tensor(bands_std)
                bands = (bands - bands_mean) / bands_std

                masks_sum = masks.sum(dim=0)
                gt_min_valid = masks_sum > self.min_valid_pixels
                fast_valid_indices = np.nonzero(gt_min_valid)

                for x in range(fast_valid_indices.shape[0]):
                    i, j = fast_valid_indices[x]
                    mask_sample = masks[:, i, j]
                    band_sample = bands[:, i, j]
                    samples = self.transform(mask_sample, band_sample, doys)
                    if len(self.buffer) < self.buffer_size:
                        self.buffer.append(samples)
                    else:
                        if self.shuffle:
                            random.shuffle(self.buffer)
                        yield from self.buffer
                        self.buffer = []

            yield from self.buffer

        return generate()

# function which returns 1) an RGB composite tile from the first tile in a SentinelTimeSeriesDataset
# and 2) a matrix of pixels which are valid for the first tile along with their positions in the composite
def get_tile(dataset):
    tile = dataset.tiles[0]
    bands = np.load(base_path + f'data/{dataset.dataset_name}/processed/{tile}/bands.npy')
    masks = np.load(base_path + f'data/{dataset.dataset_name}/processed/{tile}/masks.npy')
    bands_mean = np.load(base_path + f'data/{dataset.dataset_name}/processed/{tile}/band_mean.npy')
    bands_std = np.load(base_path + f'data/{dataset.dataset_name}/processed/{tile}/band_std.npy')
    # Convert to tensor and standardize bands
    bands = torch.tensor(bands).float()
    masks = torch.tensor(masks)
    bands_mean = torch.tensor(bands_mean)
    bands_std = torch.tensor(bands_std)
    bands = (bands - bands_mean) / bands_std

    masks_sum = masks.sum(dim=0)
    sample_size = dataset.transform.sample_size
    gt_min_valid = masks_sum > dataset.min_valid_pixels
    gt_min_valid_mask = np.nonzero(gt_min_valid)
    total_valid_samples = gt_min_valid.sum().item()

    valid_pixels = np.zeros((total_valid_samples, sample_size), dtype=np.uint8)
    valid_pixel_positions = np.zeros((total_valid_samples, 2), dtype=np.uint16)

    for i, (x, y) in enumerate(zip(gt_min_valid_mask[0], gt_min_valid_mask[1])):
        mask_sample = masks[:, x, y]
        band_sample = bands[:, x, y]
        samples, _ = dataset.transform(mask_sample, band_sample)
        valid_pixels[i] = samples[0, :, 0]
        valid_pixel_positions[i] = (x, y)

    composite_sum = np.zeros((bands.shape[1], bands.shape[2], bands.shape[3]), dtype=np.float32)
    composite_count = np.zeros((bands.shape[1], bands.shape[2], bands.shape[3]), dtype=np.float32)

    for t in range(bands.shape[0]):
        for b in [0, 1, 2]:
            band = bands[t, :, :, b]
            mask = masks[t, :, :]
            band = band.numpy()
            mask = mask.numpy()
            composite_sum[mask > 0, b] += band[mask > 0]
            composite_count[mask > 0, b] += 1

    composite = composite_sum / (composite_count + 1e-6)
    composite_rgb = np.zeros((bands.shape[1], bands.shape[2], 3), dtype=np.uint8)
    for b in range(3):
        band = composite[:, :, b]
        band = band - band.min()
        band = band / band.max()
        band = (band * 255).astype(np.uint8)

        composite_rgb[:, :, b] = band

    return composite_rgb, valid_pixels, valid_pixel_positions

# Here I shuffle all the pixels across all tiles
class SentinelTimeSeriesDatasetForDownstreaming(IterableDataset):
    def __init__(self, dataset_type, min_valid_pixels, transform, shuffle=True):
        self.min_valid_pixels = min_valid_pixels
        self.transform = transform
        self.dataset_type = dataset_type
        self.shuffle = shuffle
        self.tiles = os.listdir(base_downstream_path + f'{dataset_type}')
        self.len_cache = None

        # Store all tile data (workers will handle different tiles)
        self.all_bands = []
        self.all_masks = []
        self.all_doys = []
        self.all_labels = []

        # Find max time dimension
        self.max_time = self._find_max_time()

    def __len__(self):
        if self.len_cache is None:
            total = 0
            for tile in self.tiles:
                masks = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/masks.npy')
                masks_sum = masks.sum(axis=0)
                gt_min_valid = masks_sum > self.min_valid_pixels
                total += gt_min_valid.sum().item()

            self.len_cache = total

        return int(self.len_cache)

    def _find_max_time(self):
        """
        Find the maximum time dimension across all tiles.
        """
        max_time = 0
        for tile in self.tiles:
            bands = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/bands.npy')
            max_time = max(max_time, bands.shape[0])
        return max_time

    def _load_tile(self, tile):
        """
        Load and process a single tile, returning bands, masks, doys, and labels.
        """
        bands = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/bands.npy')
        masks = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/masks.npy')
        doys = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/doys.npy')
        labels = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/labels.npy')
        labels = np.digitize(labels, bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        bands_mean = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/band_mean.npy')
        bands_std = np.load(base_downstream_path + f'{self.dataset_type}/{tile}/band_std.npy')
        bands = (bands - bands_mean) / bands_std

        # Resize dimensions to the max time dimension
        bands = self._resize_time_dimension(bands)
        masks = self._resize_time_dimension(masks)
        doys = self._resize_day_of_year(doys)

        return bands, masks, doys, labels

    def _resize_time_dimension(self, array):
        current_time = array.shape[0]
        if current_time < self.max_time:
            pad_width = [(0, self.max_time - current_time)] + [(0, 0)] * (len(array.shape) - 1)
            array = np.pad(array, pad_width, mode='constant', constant_values=0)
        elif current_time > self.max_time:
            array = array[:self.max_time]
        return array

    def _resize_day_of_year(self, doy):
        current_length = doy.shape[0]
        if current_length < self.max_time:
            pad_width = [(0, self.max_time - current_length)]
            doy = np.pad(doy, pad_width, mode='constant', constant_values=0)
        elif current_length > self.max_time:
            doy = doy[:self.max_time]
        return doy

    def _get_worker_tiles(self):
        """
        Partition the tiles across different workers.
        """
        worker_info = get_worker_info()
        if worker_info is None:  # Single-process data loading
            return self.tiles
        
        # Split the tiles across workers
        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        per_worker_tiles = np.array_split(self.tiles, num_workers)
        
        return per_worker_tiles[worker_id]

    def __iter__(self):
        def generate():
            worker_tiles = self._get_worker_tiles()

            for tile in worker_tiles:
                bands, masks, doys, labels = self._load_tile(tile)
                total_pixels = bands.shape[1] * bands.shape[2]
                flat_indices = np.arange(total_pixels)

                if self.shuffle:
                    np.random.shuffle(flat_indices)

                for flat_index in flat_indices:
                    i = flat_index // bands.shape[2]
                    j = flat_index % bands.shape[2]

                    mask_sample = masks[:, i, j]
                    if mask_sample.sum() > self.min_valid_pixels:
                        band_sample = bands[:, i, j]
                        label_sample = labels[i, j]
                        doy_sample = doys[:]

                        # Convert to tensor
                        band_sample = torch.tensor(band_sample)
                        mask_sample = torch.tensor(mask_sample)
                        doy_sample = torch.tensor(doy_sample)
                        label_sample = torch.tensor(label_sample)

                        # Add label as padding to band sample
                        padding_tensor = torch.ones(band_sample.shape[0], 1) * label_sample
                        band_sample = torch.cat((band_sample, padding_tensor), dim=1)

                        # Apply transformation
                        sample = self.transform(mask_sample, band_sample, doy_sample)
                        yield sample

        return generate()
