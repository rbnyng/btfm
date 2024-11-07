# dataset for sentinel 2 data
# in the data/{dataset_name} directory is a directory per MGRS tile
# inside that is masks.npy and bands.npy which are (time, width, height) uint8 and (time, width, height, bands) uint16 respectively
import h5py
import torch
from torch.utils.data import IterableDataset, get_worker_info, Dataset
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, ifft
import time
from scipy.interpolate import interp1d, CubicSpline
from concurrent.futures import ThreadPoolExecutor
import logging

base_downstream_path = "../btfm-data-preparation/"

class SentinelTimeSeriesDataset(IterableDataset):
    def __init__(self, base_path, dataset_name, min_valid_pixels, transform, shuffle=True, buffer_size=6_000_00, shuffle_worker_id=None):
        self.base_path = base_path
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
                masks = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/masks.npy')
                # masks is t, h, w
                # we want to count the number of valid pixels
                masks_sum = masks.sum(axis=0)
                gt_min_valid = masks_sum > self.min_valid_pixels
                total += gt_min_valid.sum().item()

            self.len_cache = total

        return int(self.len_cache)
    
    # def get_global_mean_std(self):
    #     bands_mean = np.load(self.base_path + f'data/{self.dataset_name}/processed/band_mean.npy')

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
                bands = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/bands.npy')
                masks = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/masks.npy')
                doys = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/doys.npy')
                bands_mean = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/band_mean.npy')
                bands_std = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/band_std.npy')
                
                # test Keshav's thoughts
                # bands = np.load('/maps/sj514/btfm/data/california/processed/MGRS-10SDG/bands.npy')
                # masks = np.load('/maps/sj514/btfm/data/california/processed/MGRS-10SDG/masks.npy')
                # doys = np.load('/maps/sj514/btfm/data/california/processed/MGRS-10SDG/doys.npy')
                # bands_mean = np.load('/maps/sj514/btfm/data/california/processed/MGRS-10SDG/band_mean.npy')
                # bands_std = np.load('/maps/sj514/btfm/data/california/processed/MGRS-10SDG/band_std.npy')
                
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
                    # test Keshav's thoughts
                    # i, j = fast_valid_indices[0]
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


class SentinelTimeSeriesDatasetFixedTimestep(IterableDataset):
    def __init__(self, base_path, dataset_name, min_valid_pixels, sample_size=24, buffer_size=600_000, shuffle=True, is_training=True):
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.min_valid_pixels = min_valid_pixels
        self.sample_size = sample_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.is_training = is_training
        self.tiles = os.listdir(base_path + f'data/{dataset_name}/processed')
        self.tiles = [tile for tile in self.tiles if os.path.isdir(base_path + f'data/{dataset_name}/processed/{tile}')]
        # debug ,只取MGSR-10SGD
        # self.tiles = [tile for tile in self.tiles if tile == 'MGRS-10SGD']
        # debug, 只取前10个
        # self.tiles = self.tiles[:10]
        
        self.buffer = []
        self.len_cache = None

    def __len__(self):
        if self.len_cache is None:
            total = 0
            for tile in self.tiles:
                masks = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/masks.npy')
                masks_sum = masks.sum(axis=0)
                gt_min_valid = masks_sum > self.min_valid_pixels
                total += gt_min_valid.sum().item()
            self.len_cache = total
        return int(self.len_cache)

    def _compress_to_fixed_timesteps(self, band_sample, mask_sample, doys):
        # 压缩到固定的96个时间步
        result_bands = torch.zeros((96, band_sample.shape[1]), dtype=torch.float32)
        result_masks = torch.zeros(96, dtype=torch.int8)

        # 构建 96 个时间区间，每个区间约覆盖 3-4 天，均匀分布于 1-365 天
        week_intervals = [(i * 3.75, min(i * 3.75 + 3.75, 365)) for i in range(96)]

        for idx, (start_day, end_day) in enumerate(week_intervals):
            # 找出符合周范围的索引
            week_idx = np.where((doys >= start_day) & (doys <= end_day))[0]
            if len(week_idx) > 0:
                # 如果有多个符合，选择中间的那个
                selected_idx = week_idx[len(week_idx) // 2]
                result_bands[idx] = band_sample[selected_idx]
                result_masks[idx] = mask_sample[selected_idx]
            else:
                # 如果没有符合的，用0填充
                result_bands[idx] = 0
                result_masks[idx] = 0

        # 进行样条插值
        valid_indices = np.where(result_masks.numpy() == 1)[0]  # 找到有效的时间步
        if len(valid_indices) > 1:
            # 进行样条插值
            cs = CubicSpline(valid_indices, result_bands[valid_indices].numpy(), bc_type='natural')
            all_indices = np.arange(96)
            interpolated_bands = cs(all_indices)

            # 确保插值结果非负
            interpolated_bands = np.clip(interpolated_bands, a_min=0, a_max=None)
            result_bands = torch.tensor(interpolated_bands, dtype=torch.float32)

        return result_bands, result_masks

    def _fixed_shape_sampling(self, band_sample, valid_pixel_idx):
        sampled_band = np.zeros_like(band_sample)
        sample_mask = np.zeros(96, dtype=np.int8)

        # 选择增强方法
        if self.is_training:
            augment_method = random.choice([
                'quarterly_sampling',
                'temporal_block',
                'temporal_smoothing',
                # 'perturbation', 
                # 'frequency_domain'
            ])
        else:
            # sampled_band[valid_pixel_idx, :] = band_sample[valid_pixel_idx, :]
            sample_mask[valid_pixel_idx] = 1
            return band_sample, sample_mask

        if augment_method == 'quarterly_sampling':
            # 定义季度区间
            quarters = [
                range(0, 24),
                range(24, 48),
                range(48, 72),
                range(72, 96)
            ]
            for quarter in quarters:
                quarter_valid_idx = np.intersect1d(valid_pixel_idx, quarter)
                if len(quarter_valid_idx) > 0:
                    num_samples = min(self.sample_size // 4, len(quarter_valid_idx))
                    selected_indices = np.random.choice(quarter_valid_idx, num_samples, replace=False)
                    sampled_band[selected_indices, :] = band_sample[selected_indices, :]
                    sample_mask[selected_indices] = 1

        elif augment_method == 'temporal_smoothing':
            # Gaussian smoothing
            smoothed_band = gaussian_filter1d(band_sample, sigma=1, axis=0)
            sampled_band[valid_pixel_idx, :] = smoothed_band[valid_pixel_idx, :]
            sample_mask[valid_pixel_idx] = 1

        elif augment_method == 'perturbation':
            # Perturbation augmentation
            noise = np.random.normal(0, 0.5, band_sample[valid_pixel_idx].shape)
            sampled_band[valid_pixel_idx, :] = band_sample[valid_pixel_idx, :] + noise
            sample_mask[valid_pixel_idx] = 1
        
        elif augment_method == 'temporal_block':
            # Temporal block
            # 从0到96-sample_size中随机选择一个起始时间步
            start = np.random.choice(np.arange(96 - self.sample_size))
            selected_indices = np.arange(start, start + self.sample_size)
            band_sample[selected_indices, :] = 0
            sampled_band = band_sample
            sample_mask = np.ones(96, dtype=np.int8)
            sample_mask[selected_indices] = 0

        elif augment_method == 'frequency_domain':
            # Use FFT to retain low-frequency components
            freq_band = fft(band_sample, axis=0)  # shape: (96, 11)
            freq_band[8:] = 0  # Retain low-frequency part
            sampled_band = np.real(ifft(freq_band, axis=0))  # Inverse transform and take the real part
            sample_mask[valid_pixel_idx] = 1

        return torch.tensor(sampled_band, dtype=torch.float32), torch.tensor(sample_mask, dtype=torch.int8)

    def __iter__(self):
        def generate():
            worker_info = get_worker_info()
            if self.shuffle:
                random.shuffle(self.tiles)
            worker_tiles = self.tiles if worker_info is None else self.tiles[worker_info.id::worker_info.num_workers]

            for tile in worker_tiles:
                masks = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/masks.npy')
                # 使用 memmap 加载bands
                bands_shape = (masks.shape[0], masks.shape[1], masks.shape[2], 11)
                bands = np.memmap(self.base_path + f'data/{self.dataset_name}/processed/{tile}/bands.npy', dtype='int16', mode='r', shape=bands_shape)
                # masks = np.memmap(self.base_path + f'data/{self.dataset_name}/processed/{tile}/masks.npy', dtype='int8', mode='r', shape=(76, 1098, 1098))
                
                doys = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/doys.npy')
                bands_mean = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/band_mean.npy')
                bands_std = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/band_std.npy')
                
                # 丢弃band维度的第6个波段，对应索引为5
                bands = np.delete(bands, 5, axis=3)
                bands_mean = np.delete(bands_mean, 5)
                bands_std = np.delete(bands_std, 5)

                # 转换为 tensor
                bands = torch.tensor(bands)
                masks = torch.tensor(masks)
                doys = torch.tensor(doys, dtype=torch.int32)  # 转换为 int32 以便后续比较操作

                # 根据 min_valid_pixels 过滤
                masks_sum = masks.sum(dim=0)
                gt_min_valid = masks_sum > self.min_valid_pixels
                valid_indices = np.nonzero(gt_min_valid)

                for i, j in valid_indices:
                    # start_time = time.time()
                    mask_sample = masks[:, i, j]
                    band_sample = bands[:, i, j]
                    # standardize
                    # band_sample = (band_sample - bands_mean) / bands_std
                    doy_sample = doys

                    # 压缩时间维度到96
                    result_bands, result_masks = self._compress_to_fixed_timesteps(band_sample, mask_sample, doy_sample)

                    # 数据增强
                    final_band1, final_mask1 = self._fixed_shape_sampling(result_bands, np.nonzero(result_masks).numpy().flatten())
                    final_band2, final_mask2 = self._fixed_shape_sampling(result_bands, np.nonzero(result_masks).numpy().flatten())
                    
                    # standardize
                    final_band1 = (final_band1 - bands_mean) / bands_std
                    final_band2 = (final_band2 - bands_mean) / bands_std
                    
                    # # 转为float32
                    final_band1 = final_band1.float()
                    final_band2 = final_band2.float()
                    
                    # 缓冲输出
                    if len(self.buffer) < self.buffer_size:
                        self.buffer.append((final_band1, final_mask1, final_band2, final_mask2))
                        # end_time = time.time()
                        # print(f"Processing time for one sample: {end_time - start_time}")
                    else:
                        if self.shuffle:
                            random.shuffle(self.buffer)
                        yield from self.buffer
                        self.buffer = []

            yield from self.buffer

        return generate()


class SentinelTimeSeriesDatasetFixedTimestepForDownstreaming(IterableDataset):
    def __init__(self, base_downstream_path, dataset_type, min_valid_pixels, buffer_size=600_000, shuffle=True):
        self.base_downstream_path = base_downstream_path
        self.dataset_type = dataset_type
        self.min_valid_pixels = min_valid_pixels
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.tiles = os.listdir(base_downstream_path + f'{dataset_type}')
        # self.tiles = os.listdir(base_path + f'data/{dataset_name}/processed')
        # self.tiles = [tile for tile in self.tiles if os.path.isdir(base_path + f'data/{dataset_name}/processed/{tile}')]
        # debug ,只取MGSR-10SGD
        # self.tiles = [tile for tile in self.tiles if tile == 'MGRS-10SGD']
        # debug, 只取前10个
        # self.tiles = self.tiles[:10]
        
        # 构建 96 个时间区间，每个区间约覆盖 3-4 天，均匀分布于 1-365 天
        self.week_intervals = [(i * 3.75, min(i * 3.75 + 3.75, 365)) for i in range(96)]
        
        self.buffer = []
        self.len_cache = None

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

    def _compress_to_fixed_timesteps(self, band_sample, mask_sample, doys):
        # 压缩到固定的96个时间步
        result_bands = torch.zeros((96, band_sample.shape[1]), dtype=torch.float32)
        result_masks = torch.zeros(96, dtype=torch.int8)

        for idx, (start_day, end_day) in enumerate(self.week_intervals):
            # 找出符合周范围的索引
            week_idx = np.where((doys >= start_day) & (doys <= end_day))[0]
            if len(week_idx) > 0:
                # 如果有多个符合，选择中间的那个
                selected_idx = week_idx[len(week_idx) // 2]
                result_bands[idx] = band_sample[selected_idx]
                result_masks[idx] = mask_sample[selected_idx]
            else:
                # 如果没有符合的，用0填充
                result_bands[idx] = 0
                result_masks[idx] = 0

        # 进行样条插值
        valid_indices = np.where(result_masks.numpy() == 1)[0]  # 找到有效的时间步
        if len(valid_indices) > 1:
            # 进行样条插值
            cs = CubicSpline(valid_indices, result_bands[valid_indices].numpy(), bc_type='natural')
            all_indices = np.arange(96)
            interpolated_bands = cs(all_indices)

            # 确保插值结果非负
            interpolated_bands = np.clip(interpolated_bands, a_min=0, a_max=None)
            result_bands = torch.tensor(interpolated_bands, dtype=torch.float32)

        return result_bands, result_masks

    def __iter__(self):
        def generate():
            worker_info = get_worker_info()
            if self.shuffle:
                random.shuffle(self.tiles)
            worker_tiles = self.tiles if worker_info is None else self.tiles[worker_info.id::worker_info.num_workers]

            for tile in worker_tiles:
                doys = np.load(self.base_downstream_path + f'{self.dataset_type}/{tile}/doys.npy')
                masks = np.load(self.base_downstream_path + f'{self.dataset_type}/{tile}/masks.npy')
                # 使用 memmap 加载bands和masks
                bands_shape = (masks.shape[0], masks.shape[1], masks.shape[2], 11)
                bands = np.memmap(self.base_downstream_path + f'{self.dataset_type}/{tile}/bands.npy', dtype='int16', mode='r', shape=bands_shape)
                
                bands_mean = np.load(self.base_downstream_path + f'{self.dataset_type}/{tile}/band_mean.npy')
                bands_std = np.load(self.base_downstream_path + f'{self.dataset_type}/{tile}/band_std.npy')
                
                labels = np.load(self.base_downstream_path + f'{self.dataset_type}/{tile}/labels.npy')
                labels = np.digitize(labels, bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

                # 转换为 tensor
                bands = torch.tensor(bands)
                masks = torch.tensor(masks)
                doys = torch.tensor(doys, dtype=torch.int32)  # 转换为 int32 以便后续比较操作

                # 根据 min_valid_pixels 过滤
                masks_sum = masks.sum(dim=0)
                gt_min_valid = masks_sum > self.min_valid_pixels
                valid_indices = np.nonzero(gt_min_valid)

                for i, j in valid_indices:
                    # start_time = time.time()
                    mask_sample = masks[:, i, j]
                    band_sample = bands[:, i, j]
                    label_sample = labels[i, j]
                    # standardize
                    # band_sample = (band_sample - bands_mean) / bands_std
                    doy_sample = doys

                    # 压缩时间维度到96
                    result_bands, result_masks = self._compress_to_fixed_timesteps(band_sample, mask_sample, doy_sample)
                    
                    # standardize
                    result_bands = (result_bands - bands_mean) / bands_std
                    result_bands = result_bands.float()
                    
                    # 缓冲输出
                    if len(self.buffer) < self.buffer_size:
                        self.buffer.append((result_bands, result_masks, label_sample))
                        # end_time = time.time()
                        # print(f"Processing time for one sample: {end_time - start_time}")
                    else:
                        if self.shuffle:
                            random.shuffle(self.buffer)
                        yield from self.buffer
                        self.buffer = []

            yield from self.buffer

        return generate()

class HDF5Dataset(Dataset):
    def __init__(self, bands_file_path, masks_file_path, shuffle=True, is_training=True, start_tile_index=0, end_tile_index=1, min_valid_timesteps=16):
        self.bands_file_path = bands_file_path
        self.masks_file_path = masks_file_path
        self.shuffle = shuffle
        self.is_training = is_training
        self.min_valid_timesteps = min_valid_timesteps
        
        # Load entire datasets into memory using parallel processing
        with h5py.File(bands_file_path, 'r') as bands_file, h5py.File(masks_file_path, 'r') as masks_file:
            self.dataset_names = list(bands_file.keys())
            self.dataset_names = self.dataset_names[start_tile_index:end_tile_index]

            # Use ThreadPoolExecutor for parallel loading
            with ThreadPoolExecutor() as executor:
                bands_futures = {name: executor.submit(self._load_data, bands_file[name]['processed_bands']) for name in self.dataset_names}
                masks_futures = {name: executor.submit(self._load_data, masks_file[name]['processed_masks']) for name in self.dataset_names}

            self.bands_data = {name: future.result() for name, future in bands_futures.items()}
            self.masks_data = {name: future.result() for name, future in masks_futures.items()}

        # Create pixel indices
        height, width = self.bands_data[self.dataset_names[0]].shape[1:3]
        self.indices = np.array(np.meshgrid(range(height), range(width), indexing='ij')).T.reshape(-1, 2)

        # Create final pixels list
        num_datasets = len(self.dataset_names)
        self.pixels = np.empty((num_datasets * height * width, 3), dtype=object)

        # Fill pixels array
        for idx, dataset_name in enumerate(self.dataset_names):
            start_index = idx * height * width
            end_index = start_index + (height * width)
            self.pixels[start_index:end_index, 0] = dataset_name
            self.pixels[start_index:end_index, 1:] = self.indices

        # Shuffle pixel indices if needed
        if self.shuffle:
            np.random.shuffle(self.pixels)

    def _load_data(self, dataset):
        return dataset[:]

    def __len__(self):
        return len(self.pixels)

    def _augmentation(self, band_sample, valid_pixel_idx):
        sampled_band = np.zeros_like(band_sample)
        sample_mask = np.zeros(96, dtype=np.int8)
        
        if len(valid_pixel_idx) == 0:
            logging.warning("No valid pixel found")
            return torch.tensor(sampled_band, dtype=torch.float32), torch.tensor(sample_mask, dtype=torch.int8)

        if self.is_training:
            augment_method = random.choice([
                'interpolation', 
                'perturbation', 
                'frequency_domain',
                'random_band_dropout',
                'spectral_shift'
            ])
        else:
            sample_mask[valid_pixel_idx] = 1
            return torch.tensor(band_sample, dtype=torch.float32), torch.tensor(sample_mask, dtype=torch.int8)

        if augment_method == 'interpolation':
            valid_data = band_sample[valid_pixel_idx, :]
            for b in range(band_sample.shape[1]):
                interp_func = interp1d(valid_pixel_idx, valid_data[:, b], kind='linear', fill_value="extrapolate")
                interpolated_data = interp_func(range(96))
                interpolated_data[interpolated_data < 0] = 0
                sampled_band[:, b] = interpolated_data
            sampled_band[valid_pixel_idx, :] = valid_data
            sample_mask[valid_pixel_idx] = 1

        elif augment_method == 'perturbation':
            noise = np.random.normal(100, 20, band_sample[valid_pixel_idx].shape)
            sampled_band[valid_pixel_idx, :] = band_sample[valid_pixel_idx, :] + noise
            sample_mask[valid_pixel_idx] = 1

        elif augment_method == 'frequency_domain':
            freq_band = fft(band_sample, axis=0)
            freq_band[8:] = 0
            sampled_band = np.real(ifft(freq_band, axis=0))
            sampled_band[sampled_band < 0] = 0
            sample_mask[valid_pixel_idx] = 1

        elif augment_method == 'random_band_dropout':
            drop_indices = np.random.choice(band_sample.shape[1], size=3, replace=False)
            sampled_band = band_sample.copy()
            sampled_band[:, drop_indices] = 0
            sample_mask[valid_pixel_idx] = 1

        elif augment_method == 'spectral_shift':
            shift_value = np.random.uniform(-100, 100, band_sample.shape[1])
            sampled_band[valid_pixel_idx, :] = band_sample[valid_pixel_idx, :] + shift_value
            sampled_band[sampled_band < 0] = 0
            sample_mask[valid_pixel_idx] = 1

        return torch.tensor(sampled_band, dtype=torch.float32), torch.tensor(sample_mask, dtype=torch.int8)

    def __getitem__(self, index):
        dataset_name, i, j = self.pixels[index]
        bands = self.bands_data[dataset_name][:, i, j]
        masks = self.masks_data[dataset_name][:, i, j]
        
        # Check the sum of masks over the time dimension
        if masks.sum() < self.min_valid_timesteps:
            # Skip this item if it doesn't meet the criteria
            return self.__getitem__((index + 1) % len(self.pixels))
        
        valid_pixel_idx = np.nonzero(masks)[0]
        
        final_band1, final_mask1 = self._augmentation(bands, valid_pixel_idx)
        final_band2, final_mask2 = self._augmentation(bands, valid_pixel_idx)
        
        return final_band1, final_mask1, final_band2, final_mask2


class PastisHDF5Dataset(Dataset):
    def __init__(self, bands_file_path, masks_file_path, shuffle=True, is_training=True, start_tile_index=0, end_tile_index=1, min_valid_timesteps=16):
        self.bands_file_path = bands_file_path
        self.masks_file_path = masks_file_path
        self.shuffle = shuffle
        self.is_training = is_training
        self.min_valid_timesteps = min_valid_timesteps
        
        # Load entire datasets into memory using parallel processing
        with h5py.File(bands_file_path, 'r') as bands_file, h5py.File(masks_file_path, 'r') as masks_file:
            self.dataset_names = list(bands_file.keys())
            self.dataset_names = self.dataset_names[start_tile_index:end_tile_index]

            # Use ThreadPoolExecutor for parallel loading
            with ThreadPoolExecutor() as executor:
                bands_futures = {name: executor.submit(self._load_data, bands_file[name]) for name in self.dataset_names}
                masks_futures = {name: executor.submit(self._load_data, masks_file[name]) for name in self.dataset_names}

            self.bands_data = {name: future.result() for name, future in bands_futures.items()}
            self.masks_data = {name: future.result() for name, future in masks_futures.items()}

        # Create pixel indices
        height, width = self.bands_data[self.dataset_names[0]].shape[1:3]
        self.indices = np.array(np.meshgrid(range(height), range(width), indexing='ij')).T.reshape(-1, 2)

        # Create final pixels list
        num_datasets = len(self.dataset_names)
        self.pixels = np.empty((num_datasets * height * width, 3), dtype=object)

        # Fill pixels array
        for idx, dataset_name in enumerate(self.dataset_names):
            start_index = idx * height * width
            end_index = start_index + (height * width)
            self.pixels[start_index:end_index, 0] = dataset_name
            self.pixels[start_index:end_index, 1:] = self.indices

        # Shuffle pixel indices if needed
        if self.shuffle:
            np.random.shuffle(self.pixels)

    def _load_data(self, dataset):
        """加载HDF5数据集中的数据"""
        return dataset[:]

    def __len__(self):
        return len(self.pixels)

    def _augmentation(self, band_sample, valid_pixel_idx):
        sampled_band = np.zeros_like(band_sample)
        sample_mask = np.zeros(96, dtype=np.int8)
        
        if len(valid_pixel_idx) == 0:
            logging.warning("No valid pixel found")
            return torch.tensor(sampled_band, dtype=torch.float32), torch.tensor(sample_mask, dtype=torch.int8)

        if self.is_training:
            augment_method = random.choice([
                'interpolation', 
                'perturbation', 
                'frequency_domain',
                'random_band_dropout',
                'spectral_shift'
            ])
        else:
            sample_mask[valid_pixel_idx] = 1
            return torch.tensor(band_sample, dtype=torch.float32), torch.tensor(sample_mask, dtype=torch.int8)

        if augment_method == 'interpolation':
            valid_data = band_sample[valid_pixel_idx, :]
            for b in range(band_sample.shape[1]):
                interp_func = interp1d(valid_pixel_idx, valid_data[:, b], kind='linear', fill_value="extrapolate")
                interpolated_data = interp_func(range(96))
                interpolated_data[interpolated_data < 0] = 0
                sampled_band[:, b] = interpolated_data
            sampled_band[valid_pixel_idx, :] = valid_data
            sample_mask[valid_pixel_idx] = 1

        elif augment_method == 'perturbation':
            noise = np.random.normal(100, 20, band_sample[valid_pixel_idx].shape)
            sampled_band[valid_pixel_idx, :] = band_sample[valid_pixel_idx, :] + noise
            sample_mask[valid_pixel_idx] = 1

        elif augment_method == 'frequency_domain':
            freq_band = fft(band_sample, axis=0)
            freq_band[8:] = 0
            sampled_band = np.real(ifft(freq_band, axis=0))
            sampled_band[sampled_band < 0] = 0
            sample_mask[valid_pixel_idx] = 1

        elif augment_method == 'random_band_dropout':
            drop_indices = np.random.choice(band_sample.shape[1], size=3, replace=False)
            sampled_band = band_sample.copy()
            sampled_band[:, drop_indices] = 0
            sample_mask[valid_pixel_idx] = 1

        elif augment_method == 'spectral_shift':
            shift_value = np.random.uniform(-100, 100, band_sample.shape[1])
            sampled_band[valid_pixel_idx, :] = band_sample[valid_pixel_idx, :] + shift_value
            sampled_band[sampled_band < 0] = 0
            sample_mask[valid_pixel_idx] = 1

        return torch.tensor(sampled_band, dtype=torch.float32), torch.tensor(sample_mask, dtype=torch.int8)

    def __getitem__(self, index):
        dataset_name, i, j = self.pixels[index]
        bands = self.bands_data[dataset_name][:, i, j]
        masks = self.masks_data[dataset_name][:, i, j]
        
        # Check the sum of masks over the time dimension
        if masks.sum() < self.min_valid_timesteps:
            # Skip this item if it doesn't meet the criteria
            return self.__getitem__((index + 1) % len(self.pixels))
        
        valid_pixel_idx = np.nonzero(masks)[0]
        
        final_band1, final_mask1 = self._augmentation(bands, valid_pixel_idx)
        final_band2, final_mask2 = self._augmentation(bands, valid_pixel_idx)
        
        return final_band1, final_mask1, final_band2, final_mask2


# class SentinelTimeSeriesDatasetFixedTimestep(IterableDataset):
#     def __init__(self, base_path, dataset_name, min_valid_pixels, sample_size=16, buffer_size=600_000, shuffle=True, is_training=True):
#         self.base_path = base_path
#         self.dataset_name = dataset_name
#         self.min_valid_pixels = min_valid_pixels
#         self.sample_size = sample_size
#         self.buffer_size = buffer_size
#         self.shuffle = shuffle
#         self.is_training = is_training
#         self.tiles = os.listdir(base_path + f'data/{dataset_name}/processed')
#         self.tiles = [tile for tile in self.tiles if os.path.isdir(base_path + f'data/{dataset_name}/processed/{tile}')]
#         # self.tiles = self.tiles[:10]  # 只取前10个
        
#         self.buffer = []
#         self.len_cache = None

#     def __len__(self):
#         if self.len_cache is None:
#             total = 0
#             for tile in self.tiles:
#                 masks = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/masks.npy')
#                 masks_sum = masks.sum(axis=0)
#                 gt_min_valid = masks_sum > self.min_valid_pixels
#                 total += gt_min_valid.sum().item()
#             self.len_cache = total
#         return int(self.len_cache)

#     def _compress_to_fixed_timesteps(self, band_sample, mask_sample, doys):
#         # 压缩到固定的96个时间步
#         result_bands = torch.zeros((96, band_sample.shape[1]), dtype=torch.float32)
#         result_masks = torch.zeros(96, dtype=torch.int8)

#         # 构建 96 个时间区间
#         week_intervals = [(i * 3.75, min(i * 3.75 + 3.75, 365)) for i in range(96)]
        
#         for idx, (start_day, end_day) in enumerate(week_intervals):
#             week_idx = np.where((doys >= start_day) & (doys <= end_day))[0]
#             if len(week_idx) > 0:
#                 selected_idx = week_idx[len(week_idx) // 2]
#                 result_bands[idx] = band_sample[selected_idx]
#                 result_masks[idx] = mask_sample[selected_idx]
#             else:
#                 result_bands[idx] = 0
#                 result_masks[idx] = 0

#         # 进行样条插值
#         valid_indices = np.where(result_masks.numpy() == 1)[0]  # 找到有效的时间步
#         if len(valid_indices) > 1:
#             # 进行样条插值
#             cs = CubicSpline(valid_indices, result_bands[valid_indices].numpy(), bc_type='natural')
#             all_indices = np.arange(96)
#             result_bands = torch.tensor(cs(all_indices), dtype=torch.float32)

#         # 随机选择3个波段
#         if self.is_training and result_bands.shape[1] > 3:
#             selected_bands_indices = random.sample(range(result_bands.shape[1]), 3)
#             result_bands = result_bands[:, selected_bands_indices]
#         elif result_bands.shape[1] > 3:
#             result_bands = result_bands[:, :3]

#         return result_bands, result_masks

#     def __iter__(self):
#         def generate():
#             worker_info = get_worker_info()
#             if self.shuffle:
#                 random.shuffle(self.tiles)
#             worker_tiles = self.tiles if worker_info is None else self.tiles[worker_info.id::worker_info.num_workers]

#             for tile in worker_tiles:
#                 masks = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/masks.npy')
#                 bands_shape = (masks.shape[0], masks.shape[1], masks.shape[2], 11)
#                 bands = np.memmap(self.base_path + f'data/{self.dataset_name}/processed/{tile}/bands.npy', dtype='int16', mode='r', shape=bands_shape)
                
#                 doys = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/doys.npy')
#                 bands_mean = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/band_mean.npy')
#                 bands_std = np.load(self.base_path + f'data/{self.dataset_name}/processed/{tile}/band_std.npy')

#                 bands = torch.tensor(bands)
#                 masks = torch.tensor(masks)
#                 doys = torch.tensor(doys, dtype=torch.int32)

#                 masks_sum = masks.sum(dim=0)
#                 gt_min_valid = masks_sum > self.min_valid_pixels
#                 valid_indices = np.nonzero(gt_min_valid)

#                 for i, j in valid_indices:
#                     mask_sample = masks[:, i, j]
#                     band_sample = bands[:, i, j]

#                     # 标准化
#                     band_sample = (band_sample - bands_mean) / bands_std
#                     doy_sample = doys

#                     # 压缩时间维度到96并插值
#                     result_bands1, result_masks1 = self._compress_to_fixed_timesteps(band_sample, mask_sample, doy_sample)
#                     result_bands2, result_masks2 = self._compress_to_fixed_timesteps(band_sample, mask_sample, doy_sample)

#                     # 缓冲输出
#                     if len(self.buffer) < self.buffer_size:
#                         self.buffer.append((result_bands1, result_masks1, result_bands2, result_masks2))
#                     else:
#                         if self.shuffle:
#                             random.shuffle(self.buffer)
#                         yield from self.buffer
#                         self.buffer = []

#             yield from self.buffer

#         return generate()




if __name__ == "__main__":
    base_path = "../../sj514/btfm/"  # 替换为你的数据路径
    dataset_name = "california"  # 替换为你的数据集名称
    min_valid_pixels = 48
    sample_size = 16
    buffer_size = 6000
    shuffle = True
    is_training = True

    # 初始化数据集
    dataset = SentinelTimeSeriesDatasetFixedTimestep(
        base_path=base_path,
        dataset_name=dataset_name,
        min_valid_pixels=min_valid_pixels,
        sample_size=sample_size,
        buffer_size=buffer_size,
        shuffle=shuffle,
        is_training=is_training
    )
    # 使用 DataLoader 加载数据集
    data_loader = DataLoader(dataset, batch_size=2, num_workers=0)

    # 迭代数据集并打印每个样本的形状和内容摘要
    for idx, (bands, masks) in enumerate(data_loader):
        print(f"Sample {idx + 1}:")
        print("Bands shape:", bands.shape)  # 预期 (batch_size, 96, 11)
        print("Masks shape:", masks.shape)  # 预期 (batch_size, 96)
        print("Bands sample (first timestep):", bands[0, 0])  # 输出第一个样本的第一个时间步的数据
        print("Masks sample:", masks[0])  # 输出第一个样本的 mask
        print()

        # 为调试目的，只查看前几个批次
        if idx >= 2:
            break