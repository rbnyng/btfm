import json
import pandas as pd
import logging
import mgrs
from pyproj import Transformer
from typing import Optional, Tuple, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MGRSConverter:
    def __init__(self):
        self.mgrs_converter = mgrs.MGRS()
        self.transformers = {}
        
    def get_tile_id(self, lat: float, lon: float) -> Optional[str]:
        try:
            mgrs_id = self.mgrs_converter.toMGRS(lat, lon, MGRSPrecision=0)
            return mgrs_id
        except Exception as e:
            logging.error(f"Error converting coordinates ({lat}, {lon}) to MGRS: {str(e)}")
            return None

    def get_pixel_coordinates(self, lat: float, lon: float, mgrs_id: str) -> Optional[Tuple[int, int]]:
        try:
            # Get UTM zone from MGRS ID
            zone = int(mgrs_id[:2])
            hemisphere = 'N' if mgrs_id[2] >= 'N' else 'S'
            epsg_code = f"EPSG:{32600 + zone}" if hemisphere == 'N' else f"EPSG:{32700 + zone}"
            
            # Get or create transformer
            if epsg_code not in self.transformers:
                self.transformers[epsg_code] = Transformer.from_crs(
                    "EPSG:4326",
                    epsg_code,
                    always_xy=True
                )
            transformer = self.transformers[epsg_code]
            
            # Transform coordinates
            x_utm, y_utm = transformer.transform(lon, lat)
            
            # Calculate relative position within tile
            # Each MGRS tile is 100km x 100km
            x_offset = x_utm % 100000  # Distance from western edge of tile
            y_offset = y_utm % 100000  # Distance from southern edge of tile
            
            # Use original Sentinel-2 resolution (10m)
            PIXEL_SIZE = 10  # Original resolution
            
            # Convert to pixel coordinates at full resolution
            col = int(x_offset / PIXEL_SIZE)
            row = int((100000 - y_offset) / PIXEL_SIZE)  # Invert Y axis as image coordinates go top-down
            
            # Check bounds against full resolution (10980)
            FULL_RESOLUTION = 10980
            if 0 <= row < FULL_RESOLUTION and 0 <= col < FULL_RESOLUTION:
                logging.info(f"Successfully converted coordinates ({lat}, {lon}) to full-res pixels ({row}, {col})")
                return row, col
            else:
                logging.warning(f"Coordinates ({lat}, {lon}) fall outside tile bounds: row={row}, col={col}")
                return None
                
        except Exception as e:
            logging.error(f"Error converting coordinates ({lat}, {lon}): {str(e)}")
            return None

def generate_points_of_interest(biodiversity_df: pd.DataFrame) -> Dict[str, List[Dict[str, int]]]:
    """
    Generate points of interest JSON file from biodiversity data coordinates.
    Uses full resolution (10m) pixel coordinates.
    """
    converter = MGRSConverter()
    points_by_tile = {}
    
    skipped = 0
    processed = 0
    
    for idx, row in biodiversity_df.iterrows():
        try:
            mgrs_id = converter.get_tile_id(row['latitude'], row['longitude'])
            if mgrs_id is None:
                skipped += 1
                continue
            
            pixel_coords = converter.get_pixel_coordinates(
                row['latitude'],
                row['longitude'],
                mgrs_id
            )
            
            if pixel_coords is None:
                skipped += 1
                continue
                
            tile_id = f"MGRS-{mgrs_id}"
            row, col = pixel_coords
            
            if tile_id not in points_by_tile:
                points_by_tile[tile_id] = []
                
            points_by_tile[tile_id].append({
                "row": row,
                "col": col
            })
            
            processed += 1
            if processed % 100 == 0:
                logging.info(f"Processed {processed} points, skipped {skipped} points")
            
        except Exception as e:
            logging.error(f"Error processing row {idx}: {str(e)}")
            skipped += 1
            continue
    
    logging.info(f"Final summary: Processed {processed} points, skipped {skipped} points")
    return points_by_tile

def main():
    # Load biodiversity data
    biodiversity_df = pd.read_csv("../../../maps/ray25/data/spun_data/ECM_richness_europe.csv")
    
    # Generate points of interest
    points_by_tile = generate_points_of_interest(biodiversity_df)
    
    # Save to JSON file
    with open('points_of_interest.json', 'w') as f:
        json.dump(points_by_tile, f, indent=2)
    
    logging.info(f"Generated points of interest for {len(points_by_tile)} tiles")
    for tile_id, points in points_by_tile.items():
        logging.info(f"Tile {tile_id}: {len(points)} points")
        # Print coordinate ranges for verification
        rows = [p['row'] for p in points]
        cols = [p['col'] for p in points]
        logging.info(f"  Row range: {min(rows)}-{max(rows)}")
        logging.info(f"  Col range: {min(cols)}-{max(cols)}")

if __name__ == "__main__":
    main()