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
            pixel_data = tile_data[:, i, j, :]
            mask_data = tile_masks[:, i, j]
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

def test_model_and_visualize(model, path_to_tile, test_batch_size=512, sample_size=16, n_pca_components=3, save_dir="."):
    # Load inference tile data
    tile_bands = np.load(path_to_tile + "/bands.npy")
    tile_masks = np.load(path_to_tile + "/masks.npy")
    tile_doys = np.load(path_to_tile + "/doys.npy")

    # Extract representations from encoder
    representation_map = extract_representations(model, tile_bands, tile_masks, tile_doys, test_batch_size, sample_size)

    # Save visualization of each channel
    # for i in range(model.backbone.latent_dim):
    #     plt.imshow(representation_map[:, :, i])
    #     plt.title(f'Channel {i}')
    #     plt.axis('off')
    #     plt.imsave(f"{save_dir}/{model.backbone.__class__.__name__}_channel_{i}.png", representation_map[:, :, i])
    #     if i == 5:  # Save only a few channels for visualization
    #         break

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

    # reshape back to 3 channels
    pca_image = np.stack([r, g, b], axis=-1)

    # Visualize PCA-based representation as an image
    plt.imshow(pca_image)
    plt.title('PCA-based Representation Visualization')
    plt.axis('off')
    pca_image_path = f"{save_dir}/{model.backbone.__class__.__name__}_representation_visualization.png"
    plt.imsave(pca_image_path, pca_image)

    # Log the PCA-based representation image to WandB
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