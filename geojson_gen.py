import pandas as pd
import json
import math

def csv_to_geojson(csv_file):
    df = pd.read_csv(csv_file)
    
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    skipped_count = 0
    for idx, row in df.iterrows():
        try:
            lon = float(row['longitude'])
            lat = float(row['latitude'])
            
            if (pd.isna(lon) or pd.isna(lat) or 
                not math.isfinite(lon) or not math.isfinite(lat) or
                isinstance(lon, str) or isinstance(lat, str)):
                print(f"Skipping row {idx} due to invalid coordinates: lon={row['longitude']}, lat={row['latitude']}")
                skipped_count += 1
                continue
                
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "sample_id": row['sample_id'],
                    "continent": row['continent'],
                    "raw_obs_div": None if pd.isna(row['raw_obs_div']) else float(row['raw_obs_div']),
                    "rarefied": None if pd.isna(row['rarefied']) else float(row['rarefied'])
                }
            }
                            
            geojson["features"].append(feature)
            
        except (ValueError, TypeError) as e:
            print(f"Error processing row {idx}: {e}")
            skipped_count += 1
            continue
    
    print(f"Total rows skipped: {skipped_count}")
    return geojson

def save_geojson(geojson_data, output_file):    
    with open(output_file, 'w') as f:
        json.dump(geojson_data, f, indent=2)

if __name__ == "__main__":
    input_csv = "data\SPUN\ECM_richness_europe.csv"
    output_file = "europe.json"
    
    # Convert and save
    geojson_data = csv_to_geojson(input_csv)
    save_geojson(geojson_data, output_file)
    print(f"GeoJSON file has been created: {output_file}")