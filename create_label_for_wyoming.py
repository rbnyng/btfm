import os
import rasterio
import numpy as np
import mgrs
from pyproj import Transformer
from tqdm import tqdm

# Create UTM to WGS84 converters
utm12N_to_wgs84 = Transformer.from_crs("epsg:32612", "epsg:4326", always_xy=True)
utm13N_to_wgs84 = Transformer.from_crs("epsg:32613", "epsg:4326", always_xy=True)

# Initialize MGRS for tile-to-lat/lon conversion
m = mgrs.MGRS()

# File paths
sentinel_folder = "data-training/processed"
landcover_folder = "data-validation/worldcover"
path_to_blue_band = "data-training/wyoming/blue"

def get_tile_bounds(tile_name):
    """Calculate tile boundaries (lat/lon) based on MGRS tile name."""
    latlon_upper_left = m.toLatLon(tile_name + '00')
    latlon_lower_right = m.toLatLon(tile_name + '0909')  # 10x10 km area

    lat_min = min(latlon_upper_left[0], latlon_lower_right[0])
    lat_max = max(latlon_upper_left[0], latlon_lower_right[0])
    lon_min = min(latlon_upper_left[1], latlon_lower_right[1])
    lon_max = max(latlon_upper_left[1], latlon_lower_right[1])

    return lat_min, lat_max, lon_min, lon_max

def get_worldcover_bounds(latlon_part):
    """Calculate WorldCover file bounds (3x3 degree area) based on file name."""
    lat_dir = latlon_part[0]
    lat = int(latlon_part[1:3])
    lon_dir = latlon_part[3]
    lon = int(latlon_part[4:7])

    if lon_dir == 'W':
        lon = -lon
    if lat_dir == 'S':
        lat = -lat

    lat_min = lat
    lat_max = lat + 3
    lon_min = lon
    lon_max = lon + 3

    return lat_min, lat_max, lon_min, lon_max

def check_overlap(bounds1, bounds2):
    """Check if two bounding boxes overlap."""
    lat_min1, lat_max1, lon_min1, lon_max1 = bounds1
    lat_min2, lat_max2, lon_min2, lon_max2 = bounds2

    return not (lat_max1 < lat_min2 or lat_min1 > lat_max2 or lon_max1 < lon_min2 or lon_min1 > lon_max2)

def update_total_bounds(current_bounds, lat_min, lat_max, lon_min, lon_max):
    """Update the total geographic bounds."""
    if current_bounds is None:
        return lat_min, lat_max, lon_min, lon_max

    lat_min_total, lat_max_total, lon_min_total, lon_max_total = current_bounds
    return (
        min(lat_min_total, lat_min),
        max(lat_max_total, lat_max),
        min(lon_min_total, lon_min),
        max(lon_max_total, lon_max)
    )

def calculate_sentinel_total_bounds(sentinel_folder):
    """Calculate total geographic bounds for Sentinel-2 tiles."""
    total_bounds = None
    for file in os.listdir(sentinel_folder):
        tile_name = file.split('-')[1]  # Extract tile name
        tile_bounds = get_tile_bounds(tile_name)
        total_bounds = update_total_bounds(total_bounds, *tile_bounds)

    return total_bounds

def calculate_worldcover_total_bounds(landcover_folder):
    """Calculate total geographic bounds for WorldCover files."""
    total_bounds = None
    for file in os.listdir(landcover_folder):
        if file.endswith('.tif') and 'Map' in file:
            latlon_part = file.split('_')[5]
            worldcover_bounds = get_worldcover_bounds(latlon_part)
            total_bounds = update_total_bounds(total_bounds, *worldcover_bounds)

    return total_bounds

def match_worldcover_files(tile_bounds, landcover_folder):
    """Find matching WorldCover files for a given tile."""
    matched_files = []
    for file in tqdm(os.listdir(landcover_folder), desc="Matching LandCover files"):
        if file.endswith('.tif') and 'Map' in file:
            try:
                latlon_part = file.split('_')[5]
                worldcover_bounds = get_worldcover_bounds(latlon_part)
                if check_overlap(tile_bounds, worldcover_bounds):
                    matched_files.append(os.path.join(landcover_folder, file))
            except (IndexError, ValueError):
                print(f"Skipping file {file} due to parsing error.")
                continue

    return matched_files

def find_corresponding_landcover_pixel(UTM_number, landcover_transform, landcover_data, x, y):
    """Find corresponding Landcover pixel based on geographic coordinates."""
    if UTM_number == '12':
        lon, lat = utm12N_to_wgs84.transform(x, y)
    elif UTM_number == '13':
        lon, lat = utm13N_to_wgs84.transform(x, y)

    col, row = ~landcover_transform * (lon, lat)
    row, col = int(row), int(col)

    if 0 <= row < landcover_data.shape[0] and 0 <= col < landcover_data.shape[1]:
        return landcover_data[row, col]
    else:
        return None

def process_sentinel_file(sentinel_file, landcover_folder):
    """Process Sentinel-2 file and match it with corresponding WorldCover data."""
    tile_name = sentinel_file.split('_')[1]  # Extract tile name
    UTM_number = tile_name[0:2]

    with rasterio.open(sentinel_file) as src:
        sentinel_data = src.read(1)
        sentinel_transform = src.transform
        sentinel_data = sentinel_data[:sentinel_data.shape[0] // 10, :sentinel_data.shape[1] // 10]  # Crop to 10x10 km

    tile_bounds = get_tile_bounds(tile_name)
    matched_files = match_worldcover_files(tile_bounds, landcover_folder)
    if not matched_files:
        print(f"No matching Landcover files found for tile {tile_name}")
        return

    matched_landcover = np.zeros_like(sentinel_data, dtype=np.uint8)

    for landcover_file in matched_files:
        with rasterio.open(landcover_file) as lc_src:
            landcover_data = lc_src.read(1)
            landcover_transform = lc_src.transform

            for row in tqdm(range(sentinel_data.shape[0]), desc="Processing Sentinel rows"):
                for col in range(sentinel_data.shape[1]):
                    x, y = sentinel_transform * (col, row)
                    landcover_value = find_corresponding_landcover_pixel(UTM_number, landcover_transform, landcover_data, x, y)
                    if landcover_value is None:
                        landcover_value = 30  # Default to 30 if no match found
                    matched_landcover[row, col] = landcover_value

    output_file = os.path.join(sentinel_folder, f"MGRS-{tile_name}", "labels.tif")

    with rasterio.open(
        output_file, 'w', driver='GTiff',
        height=matched_landcover.shape[0],
        width=matched_landcover.shape[1],
        count=1, dtype=matched_landcover.dtype,
        crs=src.crs, transform=sentinel_transform
    ) as dst:
        dst.write(matched_landcover, 1)

    np.save(os.path.join(sentinel_folder, f"MGRS-{tile_name}", "labels.npy"), matched_landcover)
    print(f"Saved matched file: {output_file}")

def find_tiff_file(number, folder_path):
    """Find .tiff file in the folder based on the tile name."""
    for filename in os.listdir(folder_path):
        if filename.endswith('.tiff'):
            parts = filename.split('_')
            if len(parts) > 1 and parts[1] == number:
                return os.path.abspath(os.path.join(folder_path, filename))
    return None

def main(sentinel_folder, landcover_folder):
    """Main function to process all Sentinel-2 files."""
    sentinel_total_bounds = calculate_sentinel_total_bounds(sentinel_folder)
    worldcover_total_bounds = calculate_worldcover_total_bounds(landcover_folder)

    if check_overlap(sentinel_total_bounds, worldcover_total_bounds):
        print("Sentinel-2 and WorldCover data have overlapping geographic bounds.")
        processed_tiles = set()

        for file in os.listdir(sentinel_folder):
            tile_name = file.split('-')[1]
            if tile_name not in processed_tiles:
                sentinel_file = find_tiff_file(tile_name, path_to_blue_band)
                if sentinel_file:
                    process_sentinel_file(sentinel_file, landcover_folder)
                    processed_tiles.add(tile_name)

if __name__ == "__main__":
    main(sentinel_folder, landcover_folder)