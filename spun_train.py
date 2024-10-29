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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
        reference_file = Path(tile_path) / "blue.tiff"
        
        try:
            with rasterio.open(reference_file) as src:
                tile_crs = src.crs
                tile_transform = src.transform
                
                transformer = self._get_utm_transformer(tile_crs.to_string())
                x_utm, y_utm = transformer.transform(lon, lat)
                row, col = ~tile_transform * (x_utm, y_utm)
                
                if (0 <= row < src.height and 0 <= col < src.width):
                    return int(row), int(col)
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

class BarlowRepresentationExtractor:
    def __init__(self, checkpoint_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize backbone
        self.backbone = TransformerEncoder(
            sample_size=config["sample_size"],
            band_size=config["band_size"],
            time_dim=config["time_dim"],
            latent_dim=config["latent_dim"],
            hidden_dim=config["backbone_param_hidden_dim"],
            num_layers=config["backbone_param_num_layers"]
        )
        
        # Load trained weights
        checkpoint = torch.load(checkpoint_path)
        self.backbone.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # Create encoder model
        self.model = EncoderModel(self.backbone, config["time_dim"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Loaded Barlow Twins model from {checkpoint_path}")

    def extract_representation(self, bands, mask, doys):
        """Extract learned representation from a time series of Sentinel-2 data."""
        if mask.sum() < self.config["sample_size"]:
            return None
            
        # Sample valid timesteps
        valid_indices = np.where(mask)[0]
        selected_indices = np.random.choice(valid_indices, self.config["sample_size"], replace=False)
        
        # Extract data
        pixel_data = bands[selected_indices]
        doy_data = doys[selected_indices] // 7
        
        # Convert to tensors
        pixel_tensor = torch.tensor(pixel_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        doy_tensor = torch.tensor(doy_data, dtype=torch.long).unsqueeze(0).to(self.device)
        
        # Get representation
        with torch.no_grad():
            representation = self.model(pixel_tensor, doy_tensor)
            
        return representation.cpu().numpy()[0]

class BiodiversityPredictor:
    def __init__(self, barlow_extractor):
        self.barlow_extractor = barlow_extractor
        self.georeferencer = Sentinel2Georeferencer()
        self.rf_model = None

    def load_and_normalize_tile_data(self, tile_path):
        """Load and normalize data for a Sentinel tile."""
        try:
            bands = np.load(f"{tile_path}/bands.npy")
            masks = np.load(f"{tile_path}/masks.npy")
            doys = np.load(f"{tile_path}/doys.npy")
            
            # Load and apply normalization
            bands_mean = np.load(f"{tile_path}/band_mean.npy")
            bands_std = np.load(f"{tile_path}/band_std.npy")
            bands = (bands - bands_mean) / bands_std
            
            return bands, masks, doys
        except Exception as e:
            logging.error(f"Error loading tile data from {tile_path}: {str(e)}")
            return None, None, None
        
    def prepare_dataset(self, biodiversity_df, base_sentinel_path):
        """Prepare dataset by extracting Barlow Twins representations for each location."""
        features = []
        targets = []
        processed_locations = []
        skipped_locations = []
        
        for idx, row in tqdm(biodiversity_df.iterrows(), total=len(biodiversity_df), desc="Processing locations"):
            if pd.isna(row['rarefied']):
                skipped_locations.append((idx, "Missing biodiversity data"))
                continue
                
            # Find matching tile
            tile_path = self.georeferencer.find_matching_tile(
                row['latitude'],
                row['longitude'],
                base_sentinel_path
            )
            
            if tile_path is None:
                skipped_locations.append((idx, "No matching tile found"))
                continue
                
            # Get pixel coordinates
            pixel_coords = self.georeferencer.get_pixel_coordinates(
                row['latitude'],
                row['longitude'],
                tile_path
            )
            
            if pixel_coords is None:
                skipped_locations.append((idx, "Coordinates outside tile bounds"))
                continue
                
            # Load tile data
            bands, masks, doys = self.load_and_normalize_tile_data(tile_path)
            if bands is None:
                skipped_locations.append((idx, "Error loading tile data"))
                continue
                
            # Extract time series for this location
            row_idx, col_idx = pixel_coords
            pixel_bands = bands[:, row_idx, col_idx, :]
            pixel_mask = masks[:, row_idx, col_idx]
            
            # Get Barlow Twins representation
            representation = self.barlow_extractor.extract_representation(
                pixel_bands, pixel_mask, doys
            )
            
            if representation is not None:
                features.append(representation)
                targets.append(row['rarefied'])
                processed_locations.append({
                    'index': idx,
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'tile_path': tile_path,
                    'pixel_coords': pixel_coords
                })
            else:
                skipped_locations.append((idx, "Insufficient valid observations"))
        
        # Log processing summary
        logging.info(f"Successfully processed {len(processed_locations)} locations")
        logging.info(f"Skipped {len(skipped_locations)} locations")
        
        return np.array(features), np.array(targets), processed_locations, skipped_locations

    def train(self, X, y):
        """Train random forest using Barlow Twins representations."""
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
        
        # Evaluate
        train_pred = self.rf_model.predict(X_train)
        test_pred = self.rf_model.predict(X_test)
        
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
        
        sns.barh(data=feature_importance.tail(10), y='feature', x='importance', ax=ax2)
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
    
    # Initialize Barlow Twins representation extractor
    barlow_extractor = BarlowRepresentationExtractor(
        checkpoint_path="checkpoints/20241004_170541/model_checkpoint_val_best.pt",
        config=config
    )
    
    # Initialize biodiversity predictor
    predictor = BiodiversityPredictor(barlow_extractor)
    
    # Load biodiversity data
    biodiversity_df = pd.read_csv("AMF_richness_europe.csv")
    
    # Prepare dataset using Barlow Twins representations
    X, y, processed_locations, skipped_locations = predictor.prepare_dataset(
        biodiversity_df,
        base_sentinel_path="/maps/zf281/btfm-data-preparation/test/"
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
    logging.info(f"Training R²: {results['train_r2']:.4f}")
    logging.info(f"Test MSE: {results['test_mse']:.4f}")
    logging.info(f"Test R²: {results['test_r2']:.4f}")
    
    # Plot and save results
    fig = predictor.plot_results(results)
    fig.savefig('biodiversity_prediction_results.png')
    plt.close()

if __name__ == "__main__":
    main()
