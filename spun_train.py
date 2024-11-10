import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from tqdm import tqdm
from backbones import TransformerEncoder, TransformerEncoderWithMask
from barlow_twins import EncoderModel
import rasterio
import mgrs
from pyproj import Transformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from scipy import stats
import json
from einops import rearrange  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TileCache:
    def __init__(self, max_cache_size=5):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_order = []
    
    def get(self, tile_path):
        """Get tile data from cache or load if not present"""
        if tile_path in self.cache:
            # Update access order
            self.access_order.remove(tile_path)
            self.access_order.append(tile_path)
            return self.cache[tile_path]
            
        # Load new data
        bands, masks, doys = self._load_and_normalize_tile(tile_path)
        
        # Manage cache size
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used tile
            oldest_tile = self.access_order.pop(0)
            del self.cache[oldest_tile]
        
        # Add new data to cache
        self.cache[tile_path] = (bands, masks, doys)
        self.access_order.append(tile_path)
        
        return bands, masks, doys
    
    def _load_and_normalize_tile(self, tile_path):
        """Existing load_and_normalize_tile logic"""
        try:
            bands = np.load(tile_path / "bands.npy")
            masks = np.load(tile_path / "masks.npy")
            doys = np.load(tile_path / "doys.npy")
            
            bands = np.delete(bands, 5, axis=3)
            
            bands_mean = np.load(tile_path / "band_mean.npy")
            bands_std = np.load(tile_path / "band_std.npy")
            
            bands_mean = np.delete(bands_mean, 5)
            bands_std = np.delete(bands_std, 5)
        
            bands = (bands - bands_mean) / bands_std
            
            return bands, masks, doys
        except Exception as e:
            logging.error(f"Error loading tile data from {tile_path}: {str(e)}")
            return None, None, None
            
class RepresentationExtractor:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu', cache_size=5):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.tile_cache = TileCache(max_cache_size=cache_size)


    def extract_representation_for_coordinates(
        self,
        tile_path,
        coordinates,
        batch_size: int = 32,
        sample_size: int = 96,
        min_valid_samples: int = 32):

        bands, masks, doys = self.tile_cache.get(tile_path)
        if bands is None:
            return {}

        print(f"\nProcessing coordinates for tile: {tile_path}")
        print(f"Number of coordinates to process: {len(coordinates)}")
        print(f"First few coordinates: {coordinates[:5]}")

        representations = {}

        for row, col in coordinates:
            # Get all timestamps for this pixel
            pixel_data = bands[:, row, col, :]
            mask_data = masks[:, row, col]
            
            if mask_data.sum() < min_valid_samples:
                print(f"Skipping coordinate ({row}, {col}): insufficient valid samples")
                representations[(row, col)] = None
                continue

            # Get valid indices and sample
            valid_indices = np.nonzero(mask_data)[0]
            if len(valid_indices) < sample_size:
                selected_indices = np.random.choice(valid_indices, sample_size, replace=True)
            else:
                selected_indices = np.random.choice(valid_indices, sample_size, replace=False)
            
            # Extract samples
            sampled_pixel_data = pixel_data[selected_indices]
            
            try:
                # Convert to tensor and process
                pixel_tensor = torch.tensor(sampled_pixel_data[np.newaxis, ...], dtype=torch.float32).to(self.device)
                
                with torch.no_grad():
                    representation = self.model(pixel_tensor)
                    representations[(row, col)] = representation[0].cpu().numpy()
                    
            except Exception as e:
                logging.error(f"Error processing coordinate ({row}, {col}): {str(e)}")
                representations[(row, col)] = None

        print(f"\nProcessed tile summary:")
        print(f"Total coordinates processed: {len(coordinates)}")
        print(f"Coordinates with representations: {len([v for v in representations.values() if v is not None])}")

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



    def get_pixel_coordinates(self, lat: float, lon: float, tile_path: str) -> Optional[Tuple[int, int]]:
        """Convert lat/lon to pixel coordinates using the tile's spatial reference"""
        try:
            # Load points of interest for this tile to check sampling rate
            poi_path = Path(tile_path) / "points_of_interest.json"
            with open(poi_path, 'r') as f:
                poi_data = json.load(f) 
                poi_coords = {(p['row'], p['col']) for p in poi_data} 
                
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
            
            # Convert to full-resolution pixel coordinates first
            ORIGINAL_PIXEL_SIZE = 10  # Original Sentinel-2 resolution in meters
            full_res_col = int(x_offset / ORIGINAL_PIXEL_SIZE)
            full_res_row = int((100000 - y_offset) / ORIGINAL_PIXEL_SIZE)  # Invert Y axis
            
            # Constants matching Rust code
            REGULAR_SAMPLE_RATE = 10
            ROI_RADIUS = 5  # This should match the Rust constant
            
            # Check if this point is near any point of interest
            is_near_poi = any(
                abs(full_res_row - poi_row) <= ROI_RADIUS * 10 and  # *10 because ROI_RADIUS is in sampled units
                abs(full_res_col - poi_col) <= ROI_RADIUS * 10
                for poi_row, poi_col in poi_coords
            )
            
            # Use appropriate sampling rate
            sample_rate = 1 if is_near_poi else REGULAR_SAMPLE_RATE
            
            # Convert to sampled coordinates
            row = full_res_row // REGULAR_SAMPLE_RATE  # Always use regular sampling rate for final index
            col = full_res_col // REGULAR_SAMPLE_RATE
            
            # Check bounds
            max_size = 10980 // REGULAR_SAMPLE_RATE
            if 0 <= row < max_size and 0 <= col < max_size:
                logging.info(f"Converted coordinates ({lat}, {lon}) to pixels ({row}, {col}) using sample_rate {sample_rate}")
                return row, col
            else:
                logging.warning(f"Coordinates ({lat}, {lon}) fall outside tile bounds")
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
        self.representation_extractor = RepresentationExtractor(model)
        self.rf_model = None
        
    def prepare_dataset(self, biodiversity_df, base_sentinel_path):
        averaged_df = (biodiversity_df
            .groupby(['latitude', 'longitude'])
            .agg({
                'rarefied': 'mean'
            })
            .reset_index())
            

        available_tiles = set(Path(base_sentinel_path).glob("MGRS-*"))
        available_mgrs = {t.name.split('-')[1] for t in available_tiles}
        print(f"Available MGRS tiles: {sorted(available_mgrs)}")
        
        filtered_df = averaged_df.copy()
        filtered_df['mgrs'] = filtered_df.apply(
            lambda row: self.georeferencer.get_tile_id(row['latitude'], row['longitude']), 
            axis=1
        )
        filtered_df = filtered_df[filtered_df['mgrs'].isin(available_mgrs)]
        
        print("\nFirst few entries of filtered data:")
        print(filtered_df[['latitude', 'longitude', 'mgrs']].head())
        print("\nShape of bands.npy for first tile:")
        first_tile = next(Path(base_sentinel_path).glob("MGRS-*"))
        bands = np.load(first_tile / "bands.npy")
        print(f"Bands shape: {bands.shape}")
        print(f"Filtered from {len(biodiversity_df)} to {len(filtered_df)} locations within available tiles")
        print(f"Reduced from {len(biodiversity_df)} to {len(averaged_df)} unique coordinates after averaging")
        
        if len(filtered_df) == 0:
            raise ValueError("No biodiversity locations found within available tiles!")
        
        features = []
        targets = []
        processed_locations = []
        skipped_locations = []
        
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
        
        for tile_path, locations in tqdm(locations_by_tile.items(), desc="Processing tiles"):
            coords = [loc[2] for loc in locations]
            representations = self.representation_extractor.extract_representation_for_coordinates(
                Path(tile_path),
                coords
            )
                
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
        
        logging.info(f"Successfully processed {len(processed_locations)} locations")
        logging.info(f"Skipped {len(skipped_locations)} locations")
        
        if len(features) == 0:
            raise ValueError("No features were successfully extracted! Check your data and paths.")
            
        features_array = np.array(features)
        print("\nAll features statistics:")
        print("Shape:", features_array.shape)
        print("Mean:", np.mean(features_array))
        print("Std:", np.std(features_array))
        print("First row first 5 values:", features_array[0, :5])
        
        print("\nSample coordinates from filtered data:")
        print(filtered_df[['latitude', 'longitude', 'mgrs']].head())
        return np.array(features), np.array(targets), processed_locations, skipped_locations
        
    def save_prediction_analysis(self, y_test, test_pred, info_test, prefix=""):
        comparison_df = pd.DataFrame({
            'Latitude': [info['latitude'] for info in info_test],
            'Longitude': [info['longitude'] for info in info_test],
            'MGRS_Tile': [info['tile_path'].split('/')[-1] for info in info_test],
            'Pixel_Row': [info['pixel_coords'][0] for info in info_test],
            'Pixel_Col': [info['pixel_coords'][1] for info in info_test],
            'Actual_Biodiversity': y_test,
            'Predicted_Biodiversity': test_pred,
            'Absolute_Error': np.abs(y_test - test_pred),
            'Percent_Error': np.abs((y_test - test_pred) / y_test) * 100
        })
        
        comparison_df = comparison_df.sort_values('Absolute_Error', ascending=False)
        summary_stats = {
            'mean_absolute_error': np.mean(comparison_df['Absolute_Error']),
            'median_absolute_error': np.median(comparison_df['Absolute_Error']),
            'mean_percent_error': np.mean(comparison_df['Percent_Error']),
            'median_percent_error': np.median(comparison_df['Percent_Error']),
            'mse': mean_squared_error(y_test, test_pred),
            'r2': r2_score(y_test, test_pred)
        }
        
        filename = f"{prefix}prediction_analysis.csv"
        comparison_df.to_csv(filename, index=False)
        
        print(f"\n{prefix.title()}Set Summary Statistics:")
        print(f"Mean Absolute Error: {summary_stats['mean_absolute_error']:.2f}")
        print(f"Median Absolute Error: {summary_stats['median_absolute_error']:.2f}")
        print(f"Mean Percent Error: {summary_stats['mean_percent_error']:.2f}%")
        print(f"Median Percent Error: {summary_stats['median_percent_error']:.2f}%")
        print(f"R2 Score: {summary_stats['r2']:.4f}")
        
        print(f"\nWorst 3 Predictions ({prefix}set):")
        print(comparison_df[['Latitude', 'Longitude', 'MGRS_Tile', 
                            'Actual_Biodiversity', 'Predicted_Biodiversity', 
                            'Absolute_Error']].head(3))
        print(f"\nBest 3 Predictions ({prefix}set):")
        print(comparison_df[['Latitude', 'Longitude', 'MGRS_Tile', 
                            'Actual_Biodiversity', 'Predicted_Biodiversity', 
                            'Absolute_Error']].tail(3))
        
        return summary_stats

    def train(self, X, y, location_info):
        """Train random forest and analyze results."""
        X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
            X, y, location_info, test_size=0.2, random_state=42
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
        
        train_stats = self.save_prediction_analysis(y_train, train_pred, info_train, "train_")
        test_stats = self.save_prediction_analysis(y_test, test_pred, info_test, "test_")
        
        results = {
            'train_stats': train_stats,
            'test_stats': test_stats,
            'feature_importance': self.rf_model.feature_importances_,
            'X_test': X_test,
            'y_test': y_test,
            'test_pred': test_pred
        }
        
        return results
    
    def plot_results(self, results):
        """Plot evaluation results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.set_xlabel('Actual Biodiversity')
        ax1.set_ylabel('Predicted Biodiversity')
        ax1.set_title('Actual vs Predicted Biodiversity')
        ax1.legend()
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(results['y_test'], results['test_pred'])
        line_x = np.array([min(results['y_test']), max(results['y_test'])])
        line_y = slope * line_x + intercept

        ax1.scatter(results['y_test'], results['test_pred'], alpha=0.5, label='Predictions')
        ax1.plot(line_x, line_y, 'r--', label=f'OLS line (slope={slope:.2f})')  # OLS trend line
        ax1.plot([min(results['y_test']), max(results['y_test'])], 
                 [min(results['y_test']), max(results['y_test'])], 
                 'k:', label='Identity')  # Identity line
                 
        feature_importance = pd.DataFrame({
            'feature': [f'dim_{i}' for i in range(len(results['feature_importance']))],
            'importance': results['feature_importance']
        }).sort_values('importance', ascending=True)
        
        sns.barplot(data=feature_importance.tail(10), y='feature', x='importance', ax=ax2)
        ax2.set_title('Top 10 Most Important Representation Dimensions')
        
        plt.tight_layout()
        return fig
        
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
    
    model = TransformerEncoderWithMask()    
    
    print("\nBefore loading checkpoint:")
    print("Model state dict keys:", model.state_dict().keys())
    print("Embedding weight shape:", model.embedding.weight.shape)
    print("Embedding weight sample:", model.embedding.weight[:5, :5])
    print("Embedding on device:", model.embedding.weight.device)
    
    # Load checkpoint
    #checkpoint = torch.load("checkpoints/20241106_191719/model_checkpoint_val_best.pt")
    #checkpoint = torch.load("checkpoints/20241108_101052/model_checkpoint_step_140000.pt")
    checkpoint = torch.load("../../../maps-priv/maps/zf281/btfm-training-10.4/checkpoints/20241106_221143/model_checkpoint_val_best.pt")

    state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                 if k.startswith('backbone.')}
    
    model.load_state_dict(state_dict, strict=False)
    
    print("\nAfter loading checkpoint:")
    print("Model state dict keys:", model.state_dict().keys())
    print("Embedding weight shape:", model.embedding.weight.shape)
    print("Embedding weight sample:", model.embedding.weight[:5, :5])
    print("Embedding on device:", model.embedding.weight.device)
    
    predictor = BiodiversityPredictor(model)
    biodiversity_df = pd.read_csv("../../../maps/ray25/data/spun_data/ECM_richness_europe.csv")
    
    X, y, processed_locations, skipped_locations = predictor.prepare_dataset(
        biodiversity_df,
        base_sentinel_path="../../../maps/ray25/data/germany/processed"
    )
    
    logging.info(f"Shape of extracted representations: {X.shape}")
    
    pd.DataFrame(processed_locations).to_csv('processed_locations.csv', index=False)
    pd.DataFrame(skipped_locations, columns=['index', 'reason']).to_csv('skipped_locations.csv', index=False)
    
    results = predictor.train(X, y, processed_locations)
    
    logging.info("Model Performance:")
    logging.info(f"Training MSE: {results['train_stats']['mse']:.4f}")
    logging.info(f"Training R2: {results['train_stats']['r2']:.4f}")
    logging.info(f"Test MSE: {results['test_stats']['mse']:.4f}")
    logging.info(f"Test R2: {results['test_stats']['r2']:.4f}")
    
    fig = predictor.plot_results(results)
    fig.savefig('biodiversity_prediction_results.png')
    plt.close()

if __name__ == "__main__":
    main()