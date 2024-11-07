import os
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2
import logging
from torch import nn
from backbones import SimpleMLP, SimpleCNN, TransformerEncoder, TransformerEncoderWithMask
from multiprocessing import get_context

class EncoderModel(nn.Module):
    def __init__(self, backbone):
        super(EncoderModel, self).__init__()
        self.backbone = backbone

    def forward(self, x, mask):
        x = self.backbone(x, mask)
        return x

def extract_representations(model, bands_data, masks_data, batch_size=512):
    # Same as before
    height, width = bands_data.shape[2], bands_data.shape[3]
    representation_map = np.zeros((height, width, model.backbone.latent_dim), dtype=np.float32)

    all_pixels = []
    all_masks = []
    pixel_indices = []

    for i in range(height):
        for j in range(width):
            pixel_data = bands_data[:, :, i, j]
            mask_data = masks_data[:]

            all_pixels.append(pixel_data)
            all_masks.append(mask_data)
            pixel_indices.append((i, j))

            if len(all_pixels) == batch_size:
                batch_samples = torch.tensor(all_pixels, dtype=torch.float32).to('cuda')
                mask_samples = torch.tensor(all_masks, dtype=torch.int8).to('cuda')
                with torch.no_grad():
                    representations = model(batch_samples, mask_samples)
                for idx, (i_idx, j_idx) in enumerate(pixel_indices):
                    representation_map[i_idx, j_idx, :] = representations[idx].cpu().numpy()
                all_pixels, all_masks, pixel_indices = [], [], []

        print(f"Processed row {i+1} of {height}")

    if all_pixels:
        batch_samples = torch.tensor(all_pixels, dtype=torch.float32).to('cuda')
        mask_samples = torch.tensor(all_masks, dtype=torch.int8).to('cuda')
        with torch.no_grad():
            representations = model(batch_samples, mask_samples)
        for idx, (i_idx, j_idx) in enumerate(pixel_indices):
            representation_map[i_idx, j_idx, :] = representations[idx].cpu().numpy()

    return representation_map

def visualize_representation_map(representation_map, model_name, dataset_name, save_dir, n_pca_components=3):
    # Same as before
    height, width = representation_map.shape[:2]
    flat_representation = representation_map.reshape(-1, representation_map.shape[2])
    pca = PCA(n_components=n_pca_components)
    pca_result = pca.fit_transform(flat_representation)
    pca_image = pca_result.reshape(height, width, n_pca_components)

    r, g, b = pca_image[:, :, 0], pca_image[:, :, 1], pca_image[:, :, 2]
    r = np.clip((r - r.min()) / (r.max() - r.min()) * 255, 0, 255).astype(np.uint8)
    g = np.clip((g - g.min()) / (g.max() - g.min()) * 255, 0, 255).astype(np.uint8)
    b = np.clip((b - b.min()) / (b.max() - b.min()) * 255, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    r_clahe, g_clahe, b_clahe = clahe.apply(r), clahe.apply(g), clahe.apply(b)
    pca_image_rgb = np.stack([r_clahe, g_clahe, b_clahe], axis=-1)

    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"{model_name}_representation_{dataset_name}.png")
    plt.imsave(image_path, pca_image_rgb)
    logging.info(f"Saved PCA-based representation visualization at {image_path}")

    return pca_image_rgb

def process_dataset(dataset_name, h5_bands_path, h5_masks_path, save_dir, batch_size=512, model_path=None):
    """
    Process a single dataset and save both the representation map and its visualization.
    This function is intended to be run in parallel for each dataset.
    """
    print(f"Processing dataset {dataset_name}")
    
    # Load the model in the child process
    backbone = TransformerEncoderWithMask()
    model = EncoderModel(backbone)
    model.load_state_dict(torch.load(model_path), strict=False)
    model = model.to('cuda')
    model.eval()
    
    with h5py.File(h5_bands_path, 'r') as bands_h5, h5py.File(h5_masks_path, 'r') as masks_h5:
        bands_data = bands_h5[dataset_name][:]
        masks_data = masks_h5[dataset_name][:]

        # Extract representations
        representation_map = extract_representations(model, bands_data, masks_data, batch_size)
        
        # Save the representation map
        representation_map_path = os.path.join(save_dir, f"{dataset_name}_representation_map.npy")
        np.save(representation_map_path, representation_map)
        print(f"Saved representation map for {dataset_name} at {representation_map_path}")
        
        # Visualize and save PCA-based RGB image
        model_name = model.backbone.__class__.__name__
        visualize_representation_map(representation_map, model_name, dataset_name, save_dir)
        print(f"Visualization for {dataset_name} completed")

def process_and_visualize_all_datasets_parallel(h5_bands_path, h5_masks_path, save_dir, batch_size=512, num_workers=4, model_path=None):
    """
    Process all datasets in the HDF5 files in parallel and save both the representation maps and their visualizations.

    Args:
        h5_bands_path (str): Path to the HDF5 file containing bands data.
        h5_masks_path (str): Path to the HDF5 file containing masks data.
        save_dir (str): Directory to save output representation maps and visualizations.
        batch_size (int): Batch size for model inference.
        num_workers (int): Number of parallel workers.
        model_path (str): Path to the trained model weights.
    """
    with h5py.File(h5_bands_path, 'r') as bands_h5:
        dataset_names = list(bands_h5.keys())

    # Initialize a Pool with 'spawn' context to avoid CUDA initialization errors
    ctx = get_context("spawn")
    with ctx.Pool(processes=num_workers) as pool:
        # For each dataset, call process_dataset with necessary arguments
        pool.starmap(process_dataset, [(dataset_name, h5_bands_path, h5_masks_path, save_dir, batch_size, model_path) for dataset_name in dataset_names])

# Example usage
if __name__ == "__main__":
    MODEL_PATH = "checkpoints/transformer_shape_96_11_latent_dim_64/model_checkpoint_val_best.pt"
    H5_BANDS_PATH = "/maps/zf281/btfm-data-preparation/pastis/output_bands.h5"
    H5_MASKS_PATH = "/maps/zf281/btfm-data-preparation/pastis/output_masks.h5"
    SAVE_DIR = "pastis_representation"
    BATCH_SIZE = 64 * 64
    NUM_WORKERS = 4  # Number of parallel workers

    # Process and visualize datasets in the HDF5 files in parallel
    process_and_visualize_all_datasets_parallel(H5_BANDS_PATH, H5_MASKS_PATH, SAVE_DIR, BATCH_SIZE, NUM_WORKERS, MODEL_PATH)
