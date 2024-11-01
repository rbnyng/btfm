import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from tqdm import tqdm
from backbones import TransformerEncoder
from barlow_twins import EncoderModel
import rasterio
import mgrs
from pyproj import Transformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OptimizedRepresentationExtractor:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def load_and_normalize_tile(self, tile_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and normalize tile data."""
        try:
            # Load data
            bands = np.load(tile_path / "bands.npy")
            masks = np.load(tile_path / "masks.npy")
            doys = np.load(tile_path / "doys.npy")
            
            # Load and apply normalization
            bands_mean = np.load(tile_path / "band_mean.npy")
            bands_std = np.load(tile_path / "band_std.npy")
            bands = (bands - bands_mean) / bands_std
            
            return bands, masks, doys
        except Exception as e:
            logging.error(f"Error loading tile data from {tile_path}: {str(e)}")
            return None, None, None

    def extract_representation_for_coordinates(
        self,
        tile_path: Path,
        coordinates: List[Tuple[int, int]],
        batch_size: int = 32,
        sample_size: int = 16,
        min_valid_samples: int = 32
    ) -> Dict[Tuple[int, int], Optional[np.ndarray]]:
        """Extract representations for specific coordinates in a tile."""
        bands, masks, doys = self.load_and_normalize_tile(tile_path)
        if bands is None:
            return {}

        representations = {}
        current_batch = {
            'pixels': [],
            'doys': [],
            'coords': []
        }

        def process_batch():
            if not current_batch['pixels']:
                return
                
            batch_pixels = torch.tensor(current_batch['pixels'], dtype=torch.float32).to(self.device)
            batch_doys = torch.tensor(current_batch['doys'], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                batch_representations = self.model(batch_pixels, batch_doys)
            
            for idx, coord in enumerate(current_batch['coords']):
                representations[coord] = batch_representations[idx].cpu().numpy()
            
            current_batch['pixels'].clear()
            current_batch['doys'].clear()
            current_batch['coords'].clear()

        for row, col in coordinates:
            pixel_data = bands[:, row, col, :]
            mask_data = masks[:, row, col]
            
            if mask_data.sum() < min_valid_samples:
                representations[(row, col)] = None
                continue
                
            valid_indices = np.nonzero(mask_data)[0]
            selected_indices = np.random.choice(valid_indices, sample_size, replace=False)
            
            sampled_pixel_data = pixel_data[selected_indices]
            sampled_doy_data = doys[selected_indices] // 7
            
            current_batch['pixels'].append(sampled_pixel_data)
            current_batch['doys'].append(sampled_doy_data)
            current_batch['coords'].append((row, col))
            
            if len(current_batch['pixels']) >= batch_size:
                process_batch()
        
        process_batch()
        return representations

class Sentinel2Georeferencer:
    def __init__(self):
        self.mgrs_converter = mgrs.MGRS()
        self.transformers = {}
        
    def get_tile_id(self, lat, lon):
        try:
            mgrs_id = self.mgrs_converter.toMGRS(lat, lon, MGRSPrecision=0)
            return mgrs_id
        except Exception as e:
            logging.error(f"Error converting coordinates ({lat}, {lon}) to MGRS: {str(e)}")
            return None

    def _get_utm_transformer(self, epsg_code):
        if epsg_code not in self.transformers:
            self.transformers[epsg_code] = Transformer.from_crs(
                "EPSG:4326",
                epsg_code,
                always_xy=True
            )
        return self.transformers[epsg_code]

    def get_pixel_coordinates(self, lat, lon, tile_path):
        """Convert lat/lon to pixel coordinates using the tile's spatial reference"""
        try:
            # Load the array to get its shape
            bands = np.load(f"{tile_path}/bands.npy")
            height, width = bands.shape[1:3]
            
            # Get UTM zone from MGRS ID
            mgrs_id = Path(tile_path).name.split('-')[1]
            zone = int(mgrs_id[:2])
            hemisphere = 'N' if mgrs_id[2] >= 'N' else 'S'
            epsg_code = f"EPSG:{32600 + zone}" if hemisphere == 'N' else f"EPSG:{32700 + zone}"
            
            # Transform coordinates
            transformer = self._get_utm_transformer(epsg_code)
            x_utm, y_utm = transformer.transform(lon, lat)
            
            # Calculate relative position within tile
            # Each MGRS tile is 100km x 100km
            x_offset = x_utm % 100000  # Distance from western edge of tile
            y_offset = y_utm % 100000  # Distance from southern edge of tile
            
            # Original Sentinel-2 resolution is 10m, but we subsampled by factor of 10
            # So each pixel now represents 100m
            PIXEL_SIZE = 100  # 10m * 10 (subsampling factor)
            
            # Convert to pixel coordinates
            col = int(x_offset / PIXEL_SIZE)
            row = int((100000 - y_offset) / PIXEL_SIZE)  # Invert Y axis as image coordinates go top-down
            
            if (0 <= row < height and 0 <= col < width):
                print(f"Successfully converted coordinates ({lat}, {lon}) to pixels ({row}, {col})")
                return row, col
            else:
                logging.warning(f"Coordinates ({lat}, {lon}) fall outside tile bounds: row={row}, col={col}, height={height}, width={width}")
                return None
                
        except Exception as e:
            logging.error(f"Error converting coordinates ({lat}, {lon}): {str(e)}")
            return None
        
    def find_matching_tile(self, lat, lon, base_path):
        mgrs_id = self.get_tile_id(lat, lon)
        if mgrs_id is None:
            return None
            
        tile_path = Path(base_path) / f"MGRS-{mgrs_id}"
        
        if tile_path.exists():
            return str(tile_path)
        else:
            logging.warning(f"No tile found for MGRS ID: {mgrs_id}")
            return None

class BiodiversityPredictor:
    def __init__(self, model):
        self.model = model
        self.georeferencer = Sentinel2Georeferencer()
        self.representation_extractor = OptimizedRepresentationExtractor(model)
        self.rf_model = None
        
    def prepare_dataset(self, biodiversity_df, base_sentinel_path):
        # Get list of available tiles
        available_tiles = set(Path(base_sentinel_path).glob("MGRS-*"))
        available_mgrs = {t.name.split('-')[1] for t in available_tiles}
        print(f"Available MGRS tiles: {sorted(available_mgrs)}")
        
        # Filter biodiversity data to only include locations in available tiles
        filtered_df = biodiversity_df.copy()
        filtered_df['mgrs'] = filtered_df.apply(
            lambda row: self.georeferencer.get_tile_id(row['latitude'], row['longitude']), 
            axis=1
        )
        filtered_df = filtered_df[filtered_df['mgrs'].isin(available_mgrs)]
        # In prepare_dataset, after filtering:
        print("\nFirst few entries of filtered data:")
        print(filtered_df[['latitude', 'longitude', 'mgrs']].head())
        print("\nShape of bands.npy for first tile:")
        first_tile = next(Path(base_sentinel_path).glob("MGRS-*"))
        bands = np.load(first_tile / "bands.npy")
        print(f"Bands shape: {bands.shape}")
        print(f"Filtered from {len(biodiversity_df)} to {len(filtered_df)} locations within available tiles")
        
        if len(filtered_df) == 0:
            raise ValueError("No biodiversity locations found within available tiles!")
        
        # Process only the filtered locations
        features = []
        targets = []
        processed_locations = []
        skipped_locations = []
        
        # Group locations by tile to minimize loading/unloading tiles
        locations_by_tile = {}
        for idx, row in tqdm(filtered_df.iterrows(), desc="Processing locations"):
            if pd.isna(row['rarefied']):
                skipped_locations.append((idx, "Missing biodiversity data"))
                continue
                
            tile_path = self.georeferencer.find_matching_tile(
                row['latitude'],
                row['longitude'],
                base_sentinel_path
            )
            
            if tile_path is None:
                skipped_locations.append((idx, "No matching tile found"))
                continue
                
            pixel_coords = self.georeferencer.get_pixel_coordinates(
                row['latitude'],
                row['longitude'],
                tile_path
            )
            
            if pixel_coords is None:
                skipped_locations.append((idx, "Coordinates outside tile bounds"))
                continue
                
            if tile_path not in locations_by_tile:
                locations_by_tile[tile_path] = []
            locations_by_tile[tile_path].append((idx, row, pixel_coords))
        
        # Process each tile
        for tile_path, locations in tqdm(locations_by_tile.items(), desc="Processing tiles"):
            coords = [loc[2] for loc in locations]
            representations = self.representation_extractor.extract_representation_for_coordinates(
                Path(tile_path),
                coords
            )
            
            # Process results
            for (idx, row, coords) in locations:
                repr = representations[coords]
                if repr is not None:
                    features.append(repr)
                    targets.append(row['rarefied'])
                    processed_locations.append({
                        'index': idx,
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'tile_path': tile_path,
                        'pixel_coords': coords
                    })
                else:
                    skipped_locations.append((idx, "Insufficient valid observations"))
        
        # Log processing summary
        logging.info(f"Successfully processed {len(processed_locations)} locations")
        logging.info(f"Skipped {len(skipped_locations)} locations")
        
        if len(features) == 0:
            raise ValueError("No features were successfully extracted! Check your data and paths.")

        print("\nSample coordinates from filtered data:")
        print(filtered_df[['latitude', 'longitude', 'mgrs']].head())
        return np.array(features), np.array(targets), processed_locations, skipped_locations
    
    def train(self, X, y):
        """Train random forest using Barlow Twins representations and save detailed results."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train, y_train)
        
        train_pred = self.rf_model.predict(X_train)
        test_pred = self.rf_model.predict(X_test)
        
        # Create detailed test set comparison
        test_comparison = pd.DataFrame({
            'Actual_Biodiversity': y_test,
            'Predicted_Biodiversity': test_pred,
            'Absolute_Error': np.abs(y_test - test_pred),
            'Percent_Error': np.abs((y_test - test_pred) / y_test) * 100
        })
        
        # Sort by absolute error to easily see worst/best predictions
        test_comparison = test_comparison.sort_values('Absolute_Error', ascending=False)
        
        # Save to CSV
        test_comparison.to_csv('test_set_predictions.csv')
        
        results = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'test_r2': r2_score(y_test, test_pred),
            'feature_importance': self.rf_model.feature_importances_,
            'X_test': X_test,
            'y_test': y_test,
            'test_pred': test_pred
        }
        
        # Print summary statistics
        print("\nTest Set Summary Statistics:")
        print(f"Mean Absolute Error: {np.mean(test_comparison['Absolute_Error']):.2f}")
        print(f"Median Absolute Error: {np.median(test_comparison['Absolute_Error']):.2f}")
        print(f"Mean Percent Error: {np.mean(test_comparison['Percent_Error']):.2f}%")
        print(f"Median Percent Error: {np.median(test_comparison['Percent_Error']):.2f}%")
        print("\nWorst 3 Predictions:")
        print(test_comparison.head(3))
        print("\nBest 3 Predictions:")
        print(test_comparison.tail(3))
        
        return results
    
    def plot_results(self, results):
        """Plot evaluation results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Actual vs Predicted plot
        ax1.scatter(results['y_test'], results['test_pred'], alpha=0.5)
        ax1.plot([min(results['y_test']), max(results['y_test'])], 
                [min(results['y_test']), max(results['y_test'])], 
                'r--', label='Perfect prediction')
        ax1.set_xlabel('Actual Biodiversity')
        ax1.set_ylabel('Predicted Biodiversity')
        ax1.set_title('Actual vs Predicted Biodiversity')
        ax1.legend()
        
        # Feature Importance plot
        feature_importance = pd.DataFrame({
            'feature': [f'dim_{i}' for i in range(len(results['feature_importance']))],
            'importance': results['feature_importance']
        }).sort_values('importance', ascending=True)
        
        # Changed barh to barplot with horizontal orientation
        sns.barplot(data=feature_importance.tail(10), y='feature', x='importance', ax=ax2)
        ax2.set_title('Top 10 Most Important Representation Dimensions')
        
        plt.tight_layout()
        return fig
        
def main():
    # Configuration
    config = {
        "backbone": "transformer",
        "backbone_param_hidden_dim": 128,
        "backbone_param_num_layers": 2,
        "sample_size": 16,
        "band_size": 11,
        "latent_dim": 64,
        "time_dim": 0
    }
    
    # Initialize backbone
    backbone = TransformerEncoder(
        sample_size=config["sample_size"],
        band_size=config["band_size"],
        time_dim=config["time_dim"],
        latent_dim=config["latent_dim"],
        hidden_dim=config["backbone_param_hidden_dim"],
        num_layers=config["backbone_param_num_layers"]
    )
    
    # Load checkpoint
    checkpoint = torch.load("checkpoints/20241101_111618/model_checkpoint_val_best.pt")
    backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Create encoder model
    model = EncoderModel(backbone, config["time_dim"])
    
    # Initialize biodiversity predictor
    predictor = BiodiversityPredictor(model)
    
    # Load biodiversity data
    biodiversity_df = pd.read_csv("../../../maps/ray25/data/spun_data/ECM_richness_europe.csv")
    
    # In main(), before prepare_dataset:
    print("\nSample of biodiversity data:")
    print(biodiversity_df[['latitude', 'longitude']].describe())
    
    # Prepare dataset using Barlow Twins representations
    X, y, processed_locations, skipped_locations = predictor.prepare_dataset(
        biodiversity_df,
        base_sentinel_path="../../../maps/ray25/data/germany/processed"
    )
    
    logging.info(f"Shape of extracted representations: {X.shape}")
    
    # Save processed and skipped locations for analysis
    pd.DataFrame(processed_locations).to_csv('processed_locations.csv', index=False)
    pd.DataFrame(skipped_locations, columns=['index', 'reason']).to_csv('skipped_locations.csv', index=False)
    
    # Train and evaluate model
    results = predictor.train(X, y)
    
    # Log results
    logging.info("Model Performance:")
    logging.info(f"Training MSE: {results['train_mse']:.4f}")
    logging.info(f"Training R2: {results['train_r2']:.4f}")
    logging.info(f"Test MSE: {results['test_mse']:.4f}")
    logging.info(f"Test R2: {results['test_r2']:.4f}")
    
    # Plot and save results
    fig = predictor.plot_results(results)
    fig.savefig('biodiversity_prediction_results.png')
    plt.close()

if __name__ == "__main__":
    main()