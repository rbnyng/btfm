import torch
import numpy as np
from pathlib import Path
import json
from backbones import TransformerEncoderWithMask
from barlow_twins import EncoderModel



def get_available_tiles(base_path):
    """Get list of available MGRS tiles"""
    available_tiles = list(Path(base_path).glob("MGRS-*"))
    print(f"\nFound {len(available_tiles)} available tiles:")
    for tile in available_tiles:
        print(f"- {tile.name}")
    return available_tiles

def load_and_normalize_tile(tile_path):
    """Load and normalize a single tile's data"""
    print(f"\nLoading tile data from: {tile_path}")
    
    bands = np.load(tile_path / "bands.npy")
    masks = np.load(tile_path / "masks.npy")
    doys = np.load(tile_path / "doys.npy")
    
    print(f"Original bands shape: {bands.shape}")  # Should be [timestamps, height, width, channels]
    print(f"Masks shape: {masks.shape}")
    print(f"DOYs shape: {doys.shape}")
    
    # Remove band 5 as in original code
    bands = np.delete(bands, 5, axis=3)
    print(f"Bands shape after removing band 5: {bands.shape}")
    
    # Load and apply normalization
    bands_mean = np.delete(np.load(tile_path / "band_mean.npy"), 5)
    bands_std = np.delete(np.load(tile_path / "band_std.npy"), 5)
    
    print(f"Band means: {bands_mean}")
    print(f"Band stds: {bands_std}")
    
    bands = (bands - bands_mean) / bands_std
    
    return bands, masks, doys


def extract_single_sample(bands, masks, doys, row, col, sample_size=96):
    """Extract and process a single sample at given coordinates"""
    print(f"\nExtracting sample at coordinates: ({row}, {col})")
    
    # Get all timestamps for this pixel
    pixel_data = bands[:, row, col, :]
    mask_data = masks[:, row, col]
    
    print(f"Pixel data shape: {pixel_data.shape}")
    print(f"Valid observations: {mask_data.sum()} out of {len(mask_data)}")
    
    # Get valid indices
    valid_indices = np.nonzero(mask_data)[0]
    print(f"First few valid indices: {valid_indices[:5]}")
    print(f"Number of valid indices: {len(valid_indices)}")
    
    min_valid_samples = 32  # From original code
    if len(valid_indices) < min_valid_samples:
        print(f"Insufficient valid samples: {len(valid_indices)} < {min_valid_samples}")
        return None, None
    
    # Sample timestamps
    if len(valid_indices) < sample_size:
        print(f"Using replacement sampling as only {len(valid_indices)} valid samples available")
        selected_indices = np.random.choice(valid_indices, sample_size, replace=True)
    else:
        print(f"Sampling {sample_size} from {len(valid_indices)} valid observations")
        selected_indices = np.random.choice(valid_indices, sample_size, replace=False)
    
    # Extract samples
    sampled_pixel_data = pixel_data[selected_indices]
    sampled_doy_data = doys[selected_indices] // 7  # Convert to weeks
    
    print(f"Sampled pixel data shape: {sampled_pixel_data.shape}")  # Should be [64, 10]
    print(f"First 5 sampled DOYs (in weeks): {sampled_doy_data[:5]}")
    
    return sampled_pixel_data, sampled_doy_data

def main():
    config = {
        "backbone": "transformer",
        "backbone_param_hidden_dim": 256, 
        "backbone_param_num_layers": 6,    
        "latent_dim": 256,                
        "projection_head_hidden_dim": 128,
        "projection_head_output_dim": 128,
        "time_dim": 0,
    }

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load model
    print("\nInitializing model...")
    model = TransformerEncoderWithMask()
    
    # Load checkpoint
    checkpoint_path = "../../../maps-priv/maps/zf281/btfm-training-10.4/checkpoints/20241106_221143/model_checkpoint_val_best.pt"
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                 if k.startswith('backbone.')}
    
    print("\nCheckpoint state dict keys:")
    for k, v in state_dict.items():
        print(f"{k}: {v.shape}")
    
    model.load_state_dict(state_dict, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    # Get available tiles
    base_path = "../../../maps/ray25/data/germany/processed"
    available_tiles = get_available_tiles(base_path)
    
    if not available_tiles:
        print("No tiles found!")
        return
        
    # Process first available tile
    tile_path = available_tiles[0]
    print(f"\nProcessing tile: {tile_path}")
    
    # Load tile data
    bands, masks, doys = load_and_normalize_tile(tile_path)
    
    # Process one sample
    row, col = 574, 42  # Example coordinates
    sampled_pixel_data, sampled_doy_data = extract_single_sample(bands, masks, doys, row, col)
    
    # Convert to tensors
    print("\nPreparing input tensors...")
    pixel_tensor = torch.tensor(sampled_pixel_data[np.newaxis, ...], dtype=torch.float32).to(device)
    doy_tensor = torch.tensor(sampled_doy_data[np.newaxis, ...], dtype=torch.long).to(device)
    
    print(f"Input tensor shapes:")
    print(f"Pixel tensor: {pixel_tensor.shape}")
    print(f"DOY tensor: {doy_tensor.shape}")
    
    # Extract representation
    print("\nExtracting representation...")
    try:
        with torch.no_grad():
            representation = model(pixel_tensor, doy_tensor)
            print(f"Output representation shape: {representation.shape}")
            print(f"First 5 values of representation: {representation[0, :5].cpu().numpy()}")
    except Exception as e:
        print(f"Error during representation extraction: {str(e)}")
        print(f"Full error: {str(e.__class__.__name__)}: {str(e)}")

if __name__ == "__main__":
    main()