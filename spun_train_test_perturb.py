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
import mgrs
from pyproj import Transformer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from scipy import stats
import json
from einops import rearrange  
from scipy.interpolate import interp1d, CubicSpline
from train_new_transformer_matryoshka import MatryoshkaTransformerWithMask
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, ifft
import xgboost as xgb

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
        self.model_timesteps = 96
        
    def compress_to_fixed_timesteps(self, band_sample, mask_sample, doys):
        result_bands = torch.zeros((96, band_sample.shape[1]), dtype=torch.float32)
        result_masks = torch.zeros(96, dtype=torch.int8)

        week_intervals = [(i * 3.75, min(i * 3.75 + 3.75, 365)) for i in range(96)]

        for idx, (start_day, end_day) in enumerate(week_intervals):
            week_idx = np.where((doys >= start_day) & (doys <= end_day))[0]
            if len(week_idx) > 0:
                selected_idx = week_idx[len(week_idx) // 2]
                result_bands[idx] = band_sample[selected_idx]
                result_masks[idx] = mask_sample[selected_idx]
            else:
                result_bands[idx] = 0
                result_masks[idx] = 0

        return result_bands, result_masks
        
    def extract_representation_for_coordinates(
        self,
        tile_path,
        coordinates,
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

            try:
                sampled_pixel_data, sampled_masks = self.compress_to_fixed_timesteps(
                    pixel_data, 
                    mask_data,
                    doys
                )
                
                # Convert to tensor and process
                pixel_tensor = sampled_pixel_data.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    representation = self.model(pixel_tensor)
                    representations[(row, col)] = representation[0].cpu()
                    
            except Exception as e:
                logging.error(f"Error processing coordinate ({row}, {col}): {str(e)}")
                representations[(row, col)] = None

        print(f"\nProcessed tile summary:")
        print(f"Total coordinates processed: {len(coordinates)}")
        print(f"Coordinates with representations: {len([v for v in representations.values() if v is not None])}")

        return representations

    def extract_representation_for_coordinates_with_noise(self, tile_path, coordinates, noise_std=0.1, min_valid_samples=32, augment_method='gaussian_noise'):
        bands, masks, doys = self.tile_cache.get(tile_path)
        if bands is None:
            return {}, {}

        representations_clean = {}
        representations_augmented = {}

        for row, col in coordinates:
            pixel_data = bands[:, row, col, :]
            mask_data = masks[:, row, col]

            if mask_data.sum() < min_valid_samples:
                representations_clean[(row, col)] = None
                representations_augmented[(row, col)] = None
                continue

            try:
                pixel_tensor = torch.from_numpy(pixel_data).float()
                mask_tensor = torch.from_numpy(mask_data).int()

                sampled_pixel_data, sampled_masks = self.compress_to_fixed_timesteps(
                    pixel_tensor, mask_tensor, doys
                )

                pixel_batch = sampled_pixel_data.unsqueeze(0).to(self.device)
                mask_batch = sampled_masks.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    # Clean representation
                    all_representations_clean = self.model(pixel_batch, mask_batch)
                    representation_clean = all_representations_clean[self.dim_idx][0].cpu().numpy()

                    if augment_method == 'gaussian_noise':
                        noise = torch.randn_like(pixel_batch) * noise_std
                        augmented_pixel_tensor = pixel_batch + noise
                    elif augment_method == 'gaussian_blur':
                        augmented_pixel_tensor = torch.from_numpy(gaussian_filter1d(pixel_batch.cpu().numpy(), sigma=1, axis=0)).to(self.device)                     
                    elif augment_method == 'frequency_domain':
                        freq_band = fft(pixel_batch.cpu().numpy(), axis=1)
                        cutoff_freq = 8 
                        freq_band[:, cutoff_freq:, :] = 0
                        augmented_pixel_tensor = torch.from_numpy(np.real(ifft(freq_band, axis=1))).to(self.device)
                    elif augment_method == 'random_band_dropout':
                        drop_indices = np.random.choice(pixel_batch.shape[2], size=3, replace=False)
                        augmented_pixel_tensor = pixel_batch.clone()
                        augmented_pixel_tensor[:, :, drop_indices] = 0
                    else:
                        raise ValueError(f"Unknown augmentation method: {augment_method}")

                    all_representations_augmented = self.model(augmented_pixel_tensor, mask_batch)
                    representation_augmented = all_representations_augmented[self.dim_idx][0].cpu().numpy()

                representations_clean[(row, col)] = representation_clean
                representations_augmented[(row, col)] = representation_augmented

            except Exception as e:
                logging.error(f"Error processing coordinate ({row}, {col}): {str(e)}")
                representations_clean[(row, col)] = None
                representations_augmented[(row, col)] = None

        return representations_clean, representations_augmented
        
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
            poi_path = Path(tile_path) / "points_of_interest.json"
            poi_coords = set()  # Default to empty set
            if poi_path.exists():
                try:
                    with open(poi_path, 'r') as f:
                        poi_data = json.load(f) 
                    poi_coords = {(p['row'], p['col']) for p in poi_data} 
                except Exception as e:
                    logging.warning(f"Error loading POI data from {poi_path}: {str(e)}, using regular sampling")
                
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

class DimensionRepresentationExtractor(RepresentationExtractor):
    def __init__(self, model, dim_idx):
        super().__init__(model)
        self.dim_idx = dim_idx
        
    def extract_representation_for_coordinates(self, tile_path, coordinates, min_valid_samples=32):
        bands, masks, doys = self.tile_cache.get(tile_path)
        if bands is None:
            return {}

        representations = {}
        for row, col in coordinates:
            pixel_data = bands[:, row, col, :]
            mask_data = masks[:, row, col]
            
            if mask_data.sum() < min_valid_samples:
                representations[(row, col)] = None
                continue

            try:
                pixel_tensor = torch.from_numpy(pixel_data).float()
                mask_tensor = torch.from_numpy(mask_data).int()
        
                sampled_pixel_data, sampled_masks = self.compress_to_fixed_timesteps(
                    pixel_tensor, mask_tensor, doys
                )
                
                # Add batch dimension and move to device
                pixel_batch = sampled_pixel_data.unsqueeze(0).to(self.device)
                mask_batch = sampled_masks.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # Get all representations but only keep the one we want
                    all_representations = self.model(pixel_batch, mask_batch)
                    # Convert to numpy array for consistency with the rest of the pipeline
                    representation = all_representations[self.dim_idx][0].cpu().numpy()
                    representations[(row, col)] = representation
            
            except Exception as e:
                logging.error(f"Error processing coordinate ({row}, {col}): {str(e)}")
                representations[(row, col)] = None

        return representations


class MatryoshkaRepresentationEvaluator(BiodiversityPredictor):
    def __init__(self, model, nesting_dims=[32, 64, 128, 256, 512]):
        super().__init__(model)
        self.nesting_dims = nesting_dims
        
    def grid_search_representations(self, biodiversity_df, base_sentinel_path):
        results = {}
        
        logging.info(f"Evaluating performance across representation sizes: {self.nesting_dims}")
                        
        for idx, dim in enumerate(tqdm(self.nesting_dims, desc="Evaluating representation sizes")):
            logging.info(f"\nEvaluating dimension size: {dim}")
            
            try:
                # Set up extractor for this dimension
                self.representation_extractor = DimensionRepresentationExtractor(
                    self.model, 
                    dim_idx=idx
                )
                
                # Extract features and prepare dataset
                X, y, processed_locations, skipped_locations = self.prepare_dataset(
                    biodiversity_df,
                    base_sentinel_path
                )
                
                if len(processed_locations) < 5:
                    results[dim] = {
                        'error': f"Insufficient samples: {len(processed_locations)}"
                    }
                    continue
                    
                # Train and evaluate
                eval_results = self.train(X, y, processed_locations)
                
                # Store results
                results[dim] = {
                    'train_stats': eval_results['train_stats'],
                    'test_stats': eval_results['test_stats'],
                    'n_samples': len(y),
                    'n_skipped': len(skipped_locations),
                }
                
            except Exception as e:
                logging.error(f"Error processing dimension {dim}: {str(e)}")
                results[dim] = {'error': str(e)}
        
        return results
        
    def plot_dimension_results(self, results):
        """Plot evaluation results across different representation sizes"""
        dims = sorted([d for d in results.keys() if isinstance(d, (int, float))])
        metrics = {
            'r2': [results[d]['test_stats']['r2'] for d in dims if 'test_stats' in results[d]],
            'mse': [results[d]['test_stats']['mse'] for d in dims if 'test_stats' in results[d]],
            'mae': [results[d]['test_stats']['mean_absolute_error'] for d in dims if 'test_stats' in results[d]]
        }
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # R2 plot
        axes[0].plot(dims, metrics['r2'], 'o-')
        axes[0].set_xlabel('Representation Dimension')
        axes[0].set_ylabel('Test R2')
        axes[0].set_title('Test R2 vs Representation Size')
        axes[0].set_xscale('log', base=2)
        axes[0].grid(True)
        
        # MSE plot
        axes[1].plot(dims, metrics['mse'], 'o-')
        axes[1].set_xlabel('Representation Dimension')
        axes[1].set_ylabel('Test MSE')
        axes[1].set_title('Test MSE vs Representation Size')
        axes[1].set_xscale('log', base=2)
        axes[1].grid(True)
        
        # MAE plot
        axes[2].plot(dims, metrics['mae'], 'o-')
        axes[2].set_xlabel('Representation Dimension')
        axes[2].set_ylabel('Test MAE')
        axes[2].set_title('Test MAE vs Representation Size')
        axes[2].set_xscale('log', base=2)
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('matryoshka_dimension_results.png')
        plt.close()
        
    def analyze_perturbation_impact(self, biodiversity_df, base_sentinel_path, best_dim_idx, noise_std=0.1, augment_methods=['gaussian_noise', 'frequency_domain', 'gaussian_blur', 'random_band_dropout']):

        self.representation_extractor = RepresentationExtractor(self.model)

        for augment_method in augment_methods:
            representations_clean, representations_augmented = {}, {}

            logging.info(f"Analyzing perturbation impact with method: {augment_method}")

            for idx, row in tqdm(biodiversity_df.iterrows(), desc=f"Processing locations for {augment_method}"):
                tile_path = self.georeferencer.find_matching_tile(
                    row['latitude'], row['longitude'], base_sentinel_path
                )
                if not tile_path:
                    continue

                pixel_coords = self.georeferencer.get_pixel_coordinates(
                    row['latitude'], row['longitude'], tile_path
                )
                if not pixel_coords:
                    continue
                
                clean, augmented = self.representation_extractor.extract_representation_for_coordinates_with_noise(
                    Path(tile_path), [pixel_coords], noise_std=noise_std, augment_method=augment_method
                )

                if clean and augmented: # Check both dictionaries
                    representations_clean.update(clean)
                    representations_augmented.update(augmented)

            clean_reps = np.array([r for r in representations_clean.values() if r is not None])
            augmented_reps = np.array([r for r in representations_augmented.values() if r is not None])
            
            if len(clean_reps) == 0 or len(augmented_reps) == 0:
                raise ValueError(f"No valid representations were extracted for perturbation analysis with {augment_method}.")

            self.visualize_representations(clean_reps, augmented_reps, title=f"Perturbation Analysis ({augment_method})")

        return clean_reps, augmented_reps
        
    def visualize_representations(self, clean_reps, noisy_reps, title="Representation Visualization"):
        combined_reps = np.concatenate([clean_reps, noisy_reps])
        labels = np.array([0] * len(clean_reps) + [1] * len(noisy_reps))  # 0 for clean, 1 for noisy

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined_reps)
        self.plot_representations(pca_result, labels, title=title + " PCA")

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        tsne_result = tsne.fit_transform(combined_reps)
        self.plot_representations(tsne_result, labels, title=title + " tSNE")

    def plot_representations(self, data, labels, title):

        cmap = mcolors.ListedColormap(['blue', 'orange'])

        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, alpha=0.5, s=10)
        plt.title(title)
        plt.colorbar(label='Representation Type')
        plt.xlabel("1")
        plt.ylabel("2")
        plt.savefig(title.replace(' ', '_') + ".png")  # Save figure
        plt.close()

class MatryoshkaRepresentationEvaluatorXGB(MatryoshkaRepresentationEvaluator):
    def train(self, X, y, location_info):
        X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
            X, y, location_info, test_size=0.2, random_state=42
        )

        self.model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        train_stats = self.save_prediction_analysis(y_train, train_pred, info_train, "train_")
        test_stats = self.save_prediction_analysis(y_test, test_pred, info_test, "test_")

        results = {
            'train_stats': train_stats,
            'test_stats': test_stats,
            'feature_importance': self.model.feature_importances_,
            'X_test': X_test,
            'y_test': y_test,
            'test_pred': test_pred
        }

        return results        
        
def main_matryoshka_evaluation():
    # Initialize Matryoshka model
    model = MatryoshkaTransformerWithMask(
        input_dim=10,
        embed_dim=64, 
        num_heads=8,
        hidden_dim=256,
        num_layers=6,
        nesting_dims=[32, 64, 128, 256, 512]
    )
    
    checkpoint = torch.load("checkpoints/20241201_110529/model_checkpoint_val_best.pt")
    state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                 if k.startswith('backbone.')}
    model.load_state_dict(state_dict, strict=False)
        
    # Initialize evaluator
    evaluator = MatryoshkaRepresentationEvaluator(model)
    
    # Load biodiversity data
    biodiversity_df = pd.read_csv("../../../maps/ray25/data/spun_data/ECM_richness_europe.csv")
        
    # Perturbation Analysis
    evaluator.analyze_perturbation_impact(biodiversity_df, "../../../maps/ray25/data/germany/processed", noise_std=0.1)
    
if __name__ == "__main__":
    main_matryoshka_evaluation()