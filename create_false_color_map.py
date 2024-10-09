import os
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from backbones import SimpleMLP, SimpleCNN, TransformerEncoder

# Build the encoder (backbone only)
class EncoderModel(nn.Module):
    def __init__(self, backbone, time_dimension):
        super(EncoderModel, self).__init__()
        self.backbone = backbone
        self.time_dim = time_dimension
        # 如果为0，那么不需要time_embedding，否则为None
        self.time_embedding = None if time_dimension == 0 else nn.Embedding(53, time_dimension)

    def forward(self, x, time):
        # 如果time_dim不是0，那么将time_dim的维度加入到x中
        if self.time_embedding is not None:
            time = self.time_embedding(time)
            # 检查第一个维度是否match
            if x.size(0) != time.size(0):
                raise ValueError("Batch size of x and time should match")
            # 将两个张量在最后一个维度上拼接
            x = torch.cat([x, time], dim=-1)
        else:
            if time.size(0) != x.size(0):
                raise ValueError("Batch size of x and time should match")
        features = self.backbone(x)
        return features

run_config = {
    "backbone": "transformer",
    "backbone_param_hidden_dim": 128,
    "backbone_param_num_layers": 2,
    "min_valid": 32,
    "sample_size": 16,
    "band_size": 11,
    # "batch_size": 8192, # Maybe too large for BT, as BT doens't rely on contrastive learning.
    "batch_size": 256,
    "learning_rate": 0.00001,
    "epochs": 1,
    "latent_dim": 64,
    "validation_size": 65536*4,
    "warmup_steps": 100,
    "warmdown_steps": 250000,
    "barlow_lambda": 5e-4,
    "projection_head_hidden_dim": 128,
    "projection_head_output_dim": 128,
    "train_dataset": "california",
    "val_dataset": "california",
    "time_dim": 0,
    # "commit_link": f"https://gitlab.developers.cam.ac.uk/cst/eeg/btfm-training/-/commit/{current_git_hash}"
}

available_backbones = {
    "simple_mlp": SimpleMLP,
    "simple_cnn": SimpleCNN,
    "transformer": TransformerEncoder
}
# Load the backbone
available_backbones = {"simple_mlp": SimpleMLP, "simple_cnn": SimpleCNN, "transformer": TransformerEncoder}
# extract the backbone params from the run_config, strip the backbone_param prefix
backbone_params = {k.replace("backbone_param_", ""): v for k, v in run_config.items() if "backbone_param_" in k}

backbone = available_backbones[run_config["backbone"]](run_config["sample_size"], run_config["band_size"], run_config["time_dim"], run_config["latent_dim"], **backbone_params)
checkpoint = torch.load("checkpoints/20241004_170541/model_checkpoint_val_best.pt")
backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
# Load the trained encoder model
model = EncoderModel(backbone, run_config["time_dim"])
model.eval()
model = model.to('cuda')

# Define inference function to extract representations
def extract_representations(model, tile_data, tile_masks, tile_doys, batch_size, sample_size):
    height, width = tile_data.shape[1], tile_data.shape[2]
    representation_map = np.zeros((height, width, run_config['latent_dim']), dtype=np.float32)

    all_pixels = []
    all_doys = []
    pixel_indices = []

    for i in range(height):
        for j in range(width):
            pixel_data = tile_data[:, i, j, :]  # (time_steps, bands)
            mask_data = tile_masks[:, i, j]  # (time_steps,)
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
                batch_samples = torch.tensor(all_pixels, dtype=torch.float32).to('cuda') # torch.Size([512, 16, 11])
                doy_samples = torch.tensor(all_doys, dtype=torch.long).to('cuda') # torch.Size([512, 16])
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
        doy_samples = torch.tensor(all_doys, dtype=torch.long).to('cuda') # torch.Size([512, 16])
        with torch.no_grad():
            representations = model(batch_samples, doy_samples)
        for idx, (i_idx, j_idx) in enumerate(pixel_indices):
            representation_map[i_idx, j_idx, :] = representations[idx].cpu().numpy()

    return representation_map

# Load inference tile data
path_to_tile = "/maps/zf281/btfm-data-preparation/test/MGRS-12TYN"
tile_bands = np.load(path_to_tile + "/bands.npy")  # (73, 1098, 1098, 11)
tile_masks = np.load(path_to_tile + "/masks.npy")  # (73, 1098, 1098)
tile_doys = np.load(path_to_tile + "/doys.npy")  # (73,)

# Extract representations from encoder
representation_map = extract_representations(model, tile_bands, tile_masks, tile_doys, batch_size=512, sample_size=16)

# print the shape of the representation map
print(representation_map.shape)

# save the representation map as numpy array
save_path = os.path.join(path_to_tile, "representation_map.npy")
np.save(save_path, representation_map)

# 显示每个通道的可视化结果
for i in range(run_config['latent_dim']):
    plt.imshow(representation_map[:, :, i])
    plt.title(f'Channel {i}')
    plt.axis('off')
    plt.imsave(f"{run_config['backbone']}_channel_{i}.png", representation_map[:, :, i])
    if i == 5:
        break
    # plt.show()
# Save the representation map for later use
# np.save(f"{run_config['backbone']}_representation_map.npy", representation_map)

# Flatten representation map for PCA
height, width = representation_map.shape[:2]
flat_representation = representation_map.reshape(-1, run_config['latent_dim'])

# Apply PCA to reduce the representation to 3 dimensions (for RGB)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(flat_representation)

# Reshape PCA results back to the original tile shape
pca_image = pca_result.reshape(height, width, 3)

# 假设三个波段为rgb
r = pca_image[:, :, 0]
g = pca_image[:, :, 1]
b = pca_image[:, :, 2]

# 将三个波段变为0-255
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
plt.imsave(f"{run_config['backbone']}_representation_visualization.png", pca_image)
# plt.show()