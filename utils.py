import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
from einops import rearrange
import torch
import numpy as np
from torch import nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import wandb
from scipy.interpolate import interp1d, CubicSpline
import cv2

# def rankme(w, eps=1e-7):
#     s = w.svd(compute_uv=False)[1]
#     p = s / (s.sum() + eps)
#     return torch.exp(-(p * torch.log(p)).sum())

def rankme(z, eps=1e-7):
    s = z.svd(compute_uv=False)[1]
    p = s / (s.sum() + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    rankme_score = entropy / torch.log(torch.tensor(float(len(s))))
    return rankme_score

# generates and returns an image of the cross-correlation matrix
# for the provided z0 and z1 arrays
# z is (batch_size, latent_dim) array
def plot_cross_corr(z0, z1):
    # z0 and z1 are pytorch tensors, detach them and convert to numpy on cpu
    z0 = z0.detach().cpu().numpy()
    z1 = z1.detach().cpu().numpy()
    # normalise z0 and z1
    z0 = (z0 - z0.mean(axis=0)) / z0.std(axis=0)
    z1 = (z1 - z1.mean(axis=0)) / z1.std(axis=0)
    # compute the cross-correlation matrix
    C = np.matmul(z0.T, z1) / z0.shape[0]
    C = np.abs(C)
    # create an image to return of the cross-correlation matrix
    plt.figure()
    plt.imshow(C, cmap='binary', interpolation='nearest')
    plt.title("Embeddings cross-correlation")
    plt.colorbar()
    return plt

def extract_representations(model, tile_data, tile_masks, tile_doys, batch_size, sample_size):
    height, width = tile_data.shape[1], tile_data.shape[2]
    representation_map = np.zeros((height, width, model.backbone.latent_dim), dtype=np.float32)

    all_pixels = []
    all_doys = []
    pixel_indices = []

    for i in range(height):
        for j in range(width):
            pixel_data = tile_data[:, i, j, :] # (16, 11)
            mask_data = tile_masks[:, i, j] # (151,)
            if mask_data.sum() > 32:  # Minimum valid samples
                valid_pixel_idx = np.nonzero(mask_data)[0]
                random_idx = np.random.choice(valid_pixel_idx, sample_size, replace=False)
                pixel_data = pixel_data[random_idx]
                doy = tile_doys[random_idx] // 7
                all_pixels.append(pixel_data)
                all_doys.append(doy)
                pixel_indices.append((i, j))

            # Perform batch inference
            if len(all_pixels) == batch_size:
                batch_samples = torch.tensor(all_pixels, dtype=torch.float32).to('cuda')
                doy_samples = torch.tensor(all_doys, dtype=torch.long).to('cuda')
                with torch.no_grad():
                    representations = model(batch_samples, doy_samples)
                for idx, (i_idx, j_idx) in enumerate(pixel_indices):
                    representation_map[i_idx, j_idx, :] = representations[idx].cpu().numpy()
                all_pixels = []
                all_doys = []
                pixel_indices = []

    # Process remaining pixels
    if all_pixels:
        batch_samples = torch.tensor(all_pixels, dtype=torch.float32).to('cuda')
        doy_samples = torch.tensor(all_doys, dtype=torch.long).to('cuda')
        with torch.no_grad():
            representations = model(batch_samples, doy_samples)
        for idx, (i_idx, j_idx) in enumerate(pixel_indices):
            representation_map[i_idx, j_idx, :] = representations[idx].cpu().numpy()

    return representation_map

def extract_representations_fixed_timesteps(model, tile_data, tile_masks, doys, bands_mean, bands_std, batch_size):
    """
    Args:
        tile_data (_type_): shape: (151, 1098, 1098, 11)
        tile_masks (_type_): shape: (151, 1098, 1098)
        doys (_type_): shape: (151,)
    """
    height, width = tile_data.shape[1], tile_data.shape[2]
    representation_map = np.zeros((height, width, model.backbone.latent_dim), dtype=np.float32)

    all_pixels = []
    all_masks = []
    pixel_indices = []

    for i in range(height):
        for j in range(width):
            pixel_data = tile_data[:, i, j, :] # (151, 11)
            mask_data = tile_masks[:, i, j] # (151,)
            doy_data = doys # (151,)

            # 标准化
            # pixel_data = (pixel_data - bands_mean) / bands_std

            # 压缩时间维度到96并插值
            result_bands, result_masks = compress_to_fixed_timesteps(pixel_data, mask_data, doy_data) # (96, 3), (96,)
            
            all_pixels.append(result_bands)
            all_masks.append(result_masks)
            pixel_indices.append((i, j))
    
            # Perform batch inference
            if len(all_pixels) == batch_size:
                batch_samples = torch.stack(all_pixels, dim=0).to('cuda')
                mask_samples = torch.stack(all_masks, dim=0).to('cuda')
                with torch.no_grad():
                    representations = model(batch_samples, mask_samples)
                for idx, (i_idx, j_idx) in enumerate(pixel_indices):
                    representation_map[i_idx, j_idx, :] = representations[idx].cpu().numpy()
                all_pixels = []
                all_masks = []
                pixel_indices = []
        print(f"Processed row {i} of {height}")
                
    # Process remaining pixels
    if all_pixels:
        batch_samples = torch.stack(all_pixels, dim=0).to('cuda')
        mask_samples = torch.stack(all_masks, dim=0).to('cuda')
        with torch.no_grad():
            representations = model(batch_samples, mask_samples)
        for idx, (i_idx, j_idx) in enumerate(pixel_indices):
            representation_map[i_idx, j_idx, :] = representations[idx].cpu().numpy()

    return representation_map
            

def compress_to_fixed_timesteps(band_sample, mask_sample, doys):
        # 转为tensor
        band_sample = torch.tensor(band_sample, dtype=torch.float32)
        mask_sample = torch.tensor(mask_sample, dtype=torch.int8)
        # 压缩到固定的96个时间步
        result_bands = torch.zeros((96, band_sample.shape[1]), dtype=torch.float32)
        result_masks = torch.zeros(96, dtype=torch.int8)

        # 构建 96 个时间区间
        week_intervals = [(i * 3.75, min(i * 3.75 + 3.75, 366)) for i in range(96)]
        
        for idx, (start_day, end_day) in enumerate(week_intervals):
            week_idx = np.where((doys >= start_day) & (doys <= end_day))[0]
            if len(week_idx) > 0:
                selected_idx = week_idx[len(week_idx) // 2]
                result_bands[idx] = band_sample[selected_idx]
                result_masks[idx] = mask_sample[selected_idx]
            else:
                result_bands[idx] = 0
                result_masks[idx] = 0

        # 进行样条插值
        # valid_indices = np.where(result_masks.numpy() == 1)[0]  # 找到有效的时间步
        # if len(valid_indices) > 1:
        #     # 进行样条插值
        #     cs = CubicSpline(valid_indices, result_bands[valid_indices].numpy(), bc_type='natural')
        #     all_indices = np.arange(96)
        #     result_bands = torch.tensor(cs(all_indices), dtype=torch.float32)

        # 选择RGB波段
        # result_bands = result_bands[:, :3]

        return torch.tensor(result_bands), torch.tensor(result_masks)

def test_model_and_visualize(model, path_to_tile, test_batch_size=512, sample_size=16, n_pca_components=3, save_dir="."):
    # Load inference tile data
    tile_bands = np.load(path_to_tile + "/bands.npy")
    tile_masks = np.load(path_to_tile + "/masks.npy")
    tile_doys = np.load(path_to_tile + "/doys.npy")
    bands_mean = np.load(path_to_tile + "/band_mean.npy")
    bands_std = np.load(path_to_tile + "/band_std.npy")

    # Extract representations from encoder
    # representation_map = extract_representations(model, tile_bands, tile_masks, tile_doys, test_batch_size, sample_size)
    representation_map = extract_representations_fixed_timesteps(model, tile_bands, tile_masks, tile_doys, bands_mean, bands_std, test_batch_size)
    # save the representation map
    np.save(f"{save_dir}/{model.backbone.__class__.__name__}_representation_map.npy", representation_map)
    # Save visualization of each channel
    for i in range(model.backbone.latent_dim):
        plt.imshow(representation_map[:, :, i])
        plt.title(f'Channel {i}')
        plt.axis('off')
        plt.imsave(f"{save_dir}/{model.backbone.__class__.__name__}_channel_{i}.png", representation_map[:, :, i])
        if i == 3:  # Save only a few channels for visualization
            break

    # Flatten representation map for PCA
    height, width = representation_map.shape[:2]
    flat_representation = representation_map.reshape(-1, model.backbone.latent_dim)

    # Apply PCA to reduce the representation to n_pca_components dimensions (for visualization)
    pca = PCA(n_components=n_pca_components)
    pca_result = pca.fit_transform(flat_representation)

    # Reshape PCA results back to the original tile shape
    pca_image = pca_result.reshape(height, width, n_pca_components)

    r = pca_image[:, :, 0]
    g = pca_image[:, :, 1]
    b = pca_image[:, :, 2]

    # scale to [0, 255]
    r = np.round((r - np.min(r)) / (np.max(r) - np.min(r)) * 255)
    r = np.clip(r, 0, 255).astype(np.uint8)
    g = np.round((g - np.min(g)) / (np.max(g) - np.min(g)) * 255)
    g = np.clip(g, 0, 255).astype(np.uint8)
    b = np.round((b - np.min(b)) / (np.max(b) - np.min(b)) * 255)
    b = np.clip(b, 0, 255).astype(np.uint8)
    
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # 对每个通道应用CLAHE
    r_clahe = clahe.apply(r)
    g_clahe = clahe.apply(g)
    b_clahe = clahe.apply(b)

    # 重新组合通道
    pca_image = np.stack([r_clahe, g_clahe, b_clahe], axis=-1)

    # reshape back to 3 channels
    # pca_image = np.stack([r, g, b], axis=-1)

    # Visualize PCA-based representation as an image
    plt.imshow(pca_image)
    plt.title('PCA-based Representation Visualization')
    plt.axis('off')
    pca_image_path = f"{save_dir}/{model.backbone.__class__.__name__}_representation_visualization.png"
    plt.imsave(pca_image_path, pca_image)

    # Log the PCA-based representation image to WandB
    logging.info(f"Logging PCA-based representation visualization to WandB")
    wandb.log({"PCA-based Representation Visualization": wandb.Image(pca_image_path)})

    return pca_image


# function which returns 1) an RGB composite tile from the first tile in a SentinelTimeSeriesDataset
# and 2) a matrix of pixels which are valid for the first tile along with their positions in the composite
def get_tile(dataset, base_path):
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