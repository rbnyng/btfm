import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from data import SentinelTimeSeriesDatasetForDownstreaming
import matplotlib.pyplot as plt
from backbones import SimpleMLP, SimpleCNN, TransformerEncoder

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.fc1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.fc2(x))))
        x = self.dropout3(self.relu3(self.bn3(self.fc3(x))))
        x = self.fc4(x)
        return x

# Build the full model (backbone + classification head)
class FullModel(nn.Module):
    def __init__(self, backbone, classification_head):
        super(FullModel, self).__init__()
        self.backbone = backbone
        self.classification_head = classification_head

    def forward(self, x):
        features = self.backbone(x)
        output = self.classification_head(features)
        return output

run_config = {
    "backbone": "transformer",  # or "simple_cnn" depending on the pre-trained backbone
    "sample_size": 16,
    "band_size": 11,
    "batch_size": 512,
    "learning_rate": 0.001,
    "epochs": 10,
    "latent_dim": 128,
    "num_classes": 9  # Number of land types or classes for classification
}

available_backbones = {
    "simple_mlp": SimpleMLP,
    "simple_cnn": SimpleCNN,
    "transformer": TransformerEncoder
}
backbone = available_backbones[run_config["backbone"]](
    run_config["band_size"], run_config["latent_dim"]
)
classification_head = ClassificationHead(run_config["latent_dim"], run_config["num_classes"])

# Load the trained model
model = FullModel(backbone, classification_head)
checkpoint = torch.load("checkpoints/transformer_downstream_model_epoch_1.pth")  # Replace with your actual model checkpoint path
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to('cuda')

# Define inference function
def infer_tile(model, tile_data, tile_masks, batch_size, sample_size):
    """
    Perform inference on the given tile data and return the corresponding classification image.
    
    Args:
    - model: Pretrained model
    - tile_data: Input remote sensing data of shape (max_time, height, width, bands)
    - tile_masks: Mask data of shape (max_time, height, width)
    - batch_size: Number of samples per batch
    - sample_size: Time series length for each pixel (e.g., 16)
    
    Returns:
    - classification_map: Classification result of shape (height, width)
    """
    height, width = tile_data.shape[1], tile_data.shape[2]
    classification_map = np.zeros((height, width), dtype=np.int64)

    all_pixels = []
    pixel_indices = []

    for i in range(height):
        for j in range(width):
            pixel_data = tile_data[:, i, j, :]  # (time_steps, bands)
            mask_data = tile_masks[:, i, j]  # (time_steps,)
            if mask_data.sum() > 32:  # Minimum valid samples
                valid_pixel_idx = np.nonzero(mask_data)[0]
                random_idx = np.random.choice(valid_pixel_idx, sample_size, replace=False)
                pixel_data = pixel_data[random_idx]
                all_pixels.append(pixel_data)
                pixel_indices.append((i, j))

            # Perform batch inference
            if len(all_pixels) == batch_size:
                batch_samples = torch.tensor(all_pixels, dtype=torch.float32).to('cuda')
                with torch.no_grad():
                    outputs = model(batch_samples)
                    _, predicted = torch.max(outputs, 1)
                for idx, (i_idx, j_idx) in enumerate(pixel_indices):
                    classification_map[i_idx, j_idx] = predicted[idx].item()
                all_pixels = []
                pixel_indices = []

    # Process remaining pixels
    if all_pixels:
        batch_samples = torch.tensor(all_pixels, dtype=torch.float32).to('cuda')
        with torch.no_grad():
            outputs = model(batch_samples)
            _, predicted = torch.max(outputs, 1)
        for idx, (i_idx, j_idx) in enumerate(pixel_indices):
            classification_map[i_idx, j_idx] = predicted[idx].item()

    return classification_map

path_to_tile = "/maps/zf281/btfm-data-preparation/test/MGRS-12TYN"

# Load inference tile data
tile_bands = np.load(path_to_tile + "/bands.npy")  # (73, 1098, 1098, 11)
tile_masks = np.load(path_to_tile + "/masks.npy")  # (73, 1098, 1098)
tile_labels = np.load(path_to_tile + "/labels.npy")  # (1098, 1098)

# Convert labels to range [0, 8]
tile_labels = tile_labels // 10 - 1

# Assign colors to each label
colors = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [255, 255, 255],
    [128, 128, 128]
])
canvas = np.zeros((tile_labels.shape[0], tile_labels.shape[1], 3), dtype=np.uint8)
for i in range(9):
    canvas[tile_labels == i] = colors[i]

# Visualize labels
plt.imshow(canvas)
plt.imsave("labels.png", canvas)

# Perform inference
classification_map = infer_tile(model, tile_bands, tile_masks, batch_size=512, sample_size=16)

# Save classification map
np.save(f"{run_config['backbone']}_classification_map.npy", classification_map)

# Visualize classification result
result_canvas = np.zeros((tile_labels.shape[0], tile_labels.shape[1], 3), dtype=np.uint8)
for i in range(9):
    result_canvas[classification_map == i] = colors[i]
plt.imshow(result_canvas)
plt.imsave(f"{run_config['backbone']}_classification_map.png", result_canvas)