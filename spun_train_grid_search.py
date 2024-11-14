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
        
    def compress_to_fixed_timesteps(self, pixel_data, mask_data, doys, timestep=16):
            
        result_bands = torch.zeros((timestep, pixel_data.shape[1]), dtype=torch.float32)
        result_masks = torch.zeros(timestep, dtype=torch.int8)
        interval_length = 365 / timestep
        week_intervals = [(i * interval_length, min(i * interval_length + interval_length, 365)) 
                         for i in range(timestep)]
        
        for idx, (start_day, end_day) in enumerate(week_intervals):
            week_idx = np.where((doys >= start_day) & (doys <= end_day))[0]
            if len(week_idx) > 0:
                selected_idx = week_idx[len(week_idx) // 2]
                result_bands[idx] = torch.from_numpy(pixel_data[selected_idx])
                result_masks[idx] = mask_data[selected_idx]
        
        # Interpolation for missing values
        valid_indices = np.where(result_masks.numpy() == 1)[0]
        if len(valid_indices) > 1:
            interpolated_bands = []
            for feature_idx in range(pixel_data.shape[1]):
                cs = CubicSpline(valid_indices, 
                               result_bands[valid_indices, feature_idx].numpy(), 
                               bc_type='natural')
                all_indices = np.arange(timestep)
                interpolated = cs(all_indices)
                interpolated = np.clip(interpolated, a_min=0, a_max=None)
                interpolated_bands.append(interpolated)
            
            result_bands = torch.tensor(np.stack(interpolated_bands, axis=1), 
                                      dtype=torch.float32)
        
        if timestep is not None and timestep != self.model_timesteps:
            result_bands = self.resample_timesteps(result_bands, self.model_timesteps)
            # Create new mask for resampled data
            result_masks = torch.ones(self.model_timesteps, dtype=torch.int8)
            
        return result_bands, result_masks

    def resample_timesteps(self, data, timesteps):
        current_timesteps = data.shape[0]
        
        # Create normalized time points for current and target timesteps
        current_times = torch.linspace(0, 1, current_timesteps)
        target_times = torch.linspace(0, 1, timesteps)
        
        # Interpolate each feature dimension
        resampled_data = torch.zeros((timesteps, data.shape[1]))
        for i in range(data.shape[1]):
            resampled_data[:, i] = torch.tensor(
                np.interp(target_times.numpy(), 
                         current_times.numpy(), 
                         data[:, i].numpy())
            )
        
        return resampled_data
        
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
                # Compress to fixed timesteps and interpolate
                sampled_pixel_data, sampled_masks = self.compress_to_fixed_timesteps(
                    pixel_data, 
                    mask_data,
                    doys
                )
                
                # Convert to tensor and process
                pixel_tensor = sampled_pixel_data.unsqueeze(0).to(self.device)
                
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
        
class BiodiversityPredictorWithGridSearch(BiodiversityPredictor):
    def __init__(self, model, current_timestep=16):
        super().__init__(model)
        self.current_timestep = current_timestep
        
    def grid_search_timesteps(self, biodiversity_df, base_sentinel_path, timesteps=None):
        """
        Perform grid search over different temporal sampling rates.
        
        Args:
            biodiversity_df: DataFrame with biodiversity data
            base_sentinel_path: Path to Sentinel data
            timesteps: List of timestep values to try. If None, uses logarithmic scale.
        """
        if timesteps is None:
            # Default logarithmic scale from 1 to 96
            timesteps = [1, 2, 4, 8, 16, 32, 64, 96]
        
        results = {}
        
        for timestep in tqdm(timesteps, desc="Grid searching timesteps"):
            logging.info(f"\nEvaluating timestep: {timestep}")
            
            # Create new representation extractor with current timestep
            class TimestepRepresentationExtractor(RepresentationExtractor):
                def __init__(self, model, timestep):
                    super().__init__(model)
                    self.current_timestep = timestep
                
                def compress_to_fixed_timesteps(self, pixel_data, mask_data, doys):
                    return super().compress_to_fixed_timesteps(
                        pixel_data, mask_data, doys, 
                        timestep=self.current_timestep
                    )
            
            self.representation_extractor = TimestepRepresentationExtractor(
                self.model, 
                timestep
            )
            
            try:
                # Extract features and prepare dataset
                X, y, processed_locations, skipped_locations = self.prepare_dataset(
                    biodiversity_df,
                    base_sentinel_path
                )
                
                # Train and evaluate
                eval_results = self.train(X, y, processed_locations)
                
                # Store results
                results[timestep] = {
                    'train_stats': eval_results['train_stats'],
                    'test_stats': eval_results['test_stats'],
                    'n_samples': len(y),
                    'n_skipped': len(skipped_locations)
                }
                
                # Save intermediate results
                self.save_timestep_results(results, f'timestep_results_{timestep}.json')
                
            except Exception as e:
                logging.error(f"Error processing timestep {timestep}: {str(e)}")
                results[timestep] = {'error': str(e)}
        
        return results
    
    def save_timestep_results(self, results, filename):
        """Save results to JSON file"""
        # Convert results to serializable format
        serializable_results = {}
        for timestep, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                serializable_results[str(timestep)] = result
            else:
                serializable_results[str(timestep)] = {
                    'train_stats': result['train_stats'],
                    'test_stats': result['test_stats'],
                    'n_samples': result['n_samples'],
                    'n_skipped': result['n_skipped']
                }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def plot_grid_search_results(self, results):
        """Plot grid search results"""
        timesteps = sorted([t for t in results.keys() if isinstance(t, (int, float))])
        test_r2 = [results[t]['test_stats']['r2'] for t in timesteps 
                   if 'test_stats' in results[t]]
        test_mse = [results[t]['test_stats']['mse'] for t in timesteps 
                   if 'test_stats' in results[t]]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # R2 plot
        ax1.plot(timesteps, test_r2, 'o-')
        ax1.set_xlabel('Number of Timesteps')
        ax1.set_ylabel('Test R2')
        ax1.set_title('Test R2 vs Number of Timesteps')
        ax1.set_xscale('log')
        ax1.grid(True)
        
        # MSE plot
        ax2.plot(timesteps, test_mse, 'o-')
        ax2.set_xlabel('Number of Timesteps')
        ax2.set_ylabel('Test MSE')
        ax2.set_title('Test MSE vs Number of Timesteps')
        ax2.set_xscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('timestep_grid_search_results.png')
        plt.close()

def main_grid_search():
    # Initialize model same as before
    model = TransformerEncoderWithMask()
    checkpoint = torch.load("checkpoints/20241111_200219/model_checkpoint_step_20000.pt")
    state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                 if k.startswith('backbone.')}
    model.load_state_dict(state_dict, strict=False)
    
    # Initialize predictor with grid search
    predictor = BiodiversityPredictorWithGridSearch(model)
    biodiversity_df = pd.read_csv("../../../maps/ray25/data/spun_data/ECM_richness_europe.csv")
    
    # Define timesteps for grid search
    timesteps = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96]
    
    # Run grid search
    results = predictor.grid_search_timesteps(
        biodiversity_df,
        base_sentinel_path="../../../maps/ray25/data/germany/processed",
        timesteps=timesteps
    )
    
    # Save final results
    predictor.save_timestep_results(results, 'timestep_grid_search_final.json')
    
    # Plot results
    predictor.plot_grid_search_results(results)
    
    # Print best timestep
    test_r2_scores = {t: results[t]['test_stats']['r2'] for t in timesteps 
                      if 'test_stats' in results[t]}
    best_timestep = max(test_r2_scores.items(), key=lambda x: x[1])[0]
    logging.info(f"\nBest timestep: {best_timestep}")
    logging.info(f"Best R2 score: {test_r2_scores[best_timestep]:.4f}")

if __name__ == "__main__":
    main_grid_search()