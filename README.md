# Btfm training

Processing and analyzing Sentinel-2 satellite imagery data using the Barlow Twins approach. 

The resulting foundation model can be fine-tuned for downstream tasks.

## Components

- **Data Handling**: Dataset classes for Sentinel-2 time series data (`data.py`)
- **Model Architectures**: Various options including MLP, CNN, and transformer (`backbones.py`)
- **Self-Supervised Learning**: Implementation of Barlow Twins (`barlow_twins.py`)
- **Training Scripts**: 
  - Self-supervised training (`train.py`)
  - Supervised fine-tuning for classification (`train_classification.py`)
- **Inference and Visualization**: Tools for model inference and creating false-color maps (`infer.py`, `create_false_color_map.py`)
- **Utilities**: Helper functions for analysis and visualization (`utils.py`)

## Pipeline

1. **Train Barlow Twins on Sentinel-2 data**
   - Script: `train.py`
   - This script implements the Barlow Twins SSL approach.
   - It uses the `SentinelTimeSeriesDataset` class from `data.py` to load and process Sentinel-2 time series data.
   - The model architecture (e.g., MLP, CNN, or transformer) is defined in `backbones.py`.
   - The Barlow Twins specific components (loss function, projection head) are implemented in `barlow_twins.py`.
   - Training progress and metrics are logged using wandb.

2. **Finetune the pre trained model for land cover classification**
   - Scripts: `train_classification.py`
   - These scripts load the pre-trained model from step 1 and fine-tune it for land cover classification.
   - They use the `SentinelTimeSeriesDatasetForDownstreaming` class from `data.py` for loading labeled data.
   - A classification head is added on top of the pre-trained backbone.
   - `train_classification.py` uses a more complex training setup with learning rate scheduling and validation.
   - `train_classification_maddy.py` provides a simpler training loop with Focal Loss option.

3. **Inference on new data and generate visualizations**
   - Scripts: `infer.py` and `create_false_color_map.py`
   - `infer.py` loads the fine-tuned model and runs inference on new Sentinel-2 data.
   - It uses functions from `utils.py` for extracting representations and processing results.
   - `create_false_color_map.py` generates false-color visualizations of the model's representations.
   - The script `read_inference_map.py` can be used to load and display the generated classification maps.

Utilities:
- `transforms.py` defines data augmentation and preprocessing transforms used in training.
- `utils.py` contains helper functions for various tasks such as computing effective rank and plotting cross-correlation matrices.

To run the complete workflow:
1. Execute `train.py` to obtain a pre-trained model.
2. Run `train_classification.py` for fine-tuning.
3. Use `infer.py` followed by `create_false_color_map.py` to generate visualizations of results on new data.

## File Paths Config

### Base Path
- **Variable**: `base_path` in `SentinelTimeSeriesDataset` class (`data.py`)
- **Description**: This is the root directory of the project. It should contain the `data` folder and the scripts files.

### Dataset Path
- **Variable**: `dataset_name` in `run_config` (`train.py`)
- **Description**: This is the name of the subdirectory under `data/` that contains processed sentinel-2 tiles.

### Test Tile Path
- **Variable**: `"test_tile_path"` in `run_config` (`train.py`)
- **Description**: This is the path to a specific tile used for testing and visualization.

### Data Structure
The data should be organized as follows:

```
/base_path/
└── data/
    └── dataset_name/
        └── processed/
            └── MGRS-XXXXXX/
                ├── bands.npy
                ├── masks.npy
                ├── doys.npy
                ├── band_mean.npy
                └── band_std.npy
```