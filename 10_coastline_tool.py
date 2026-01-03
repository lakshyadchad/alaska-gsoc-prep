import argparse
import json
import os
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

def load_image(filepath):
    """Loads the satellite image and returns bands + metadata."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with rasterio.open(filepath) as src:
        data = {
            'green': src.read(2),
            'nir': src.read(4),
            'mask': src.read_masks(1),
            'transform': src.transform,
            'crs': src.crs,
            'width': src.width,
            'height': src.height
        }
    return data

def calculate_water_mask(data):
    """Generates a binary water mask using NDWI + Otsu + LCC Cleaning."""
    print("   ...Calculating NDWI")
    green = data['green'].astype(float)
    nir = data['nir'].astype(float)
    
    # NDWI Calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi = (green - nir) / (green + nir + 0.00001)
    
    # Filter valid pixels for Otsu
    valid_pixels = ndwi[data['mask'] == 255]
    valid_pixels = valid_pixels[~np.isnan(valid_pixels)]
    
    if len(valid_pixels) == 0:
        raise ValueError("No valid data pixels found in image.")

    # Adaptive Thresholding
    otsu_thresh = threshold_otsu(valid_pixels)
    print(f"   ...Otsu Threshold Calculated: {otsu_thresh:.4f}")
    
    raw_mask = (ndwi > otsu_thresh) & (data['mask'] == 255)
    
    # Largest Connected Component (LCC) Cleaning
    print("   ...Applying Area Filtering (Keep Largest)")
    lbl_matrix = label(raw_mask)
    regions = regionprops(lbl_matrix)
    
    if not regions:
        print("   WARNING: No water detected.")
        return np.zeros_like(raw_mask, dtype=np.uint8)

    largest = max(regions, key=lambda r: r.area)
    clean_mask = np.zeros_like(raw_mask, dtype=np.uint8)
    clean_mask[lbl_matrix == largest.label] = 1
    
    return clean_mask

def vectorize_mask(binary_mask, transform):
    """Converts binary mask to GeoJSON-compatible polygons."""
    print("   ...Vectorizing")
    results = (
        {'properties': {'type': 'water'}, 'geometry': s}
        for i, (s, v) in enumerate(shapes(binary_mask, mask=binary_mask.astype(bool), transform=transform))
        if v == 1
    )
    # Convert to Shapely to fix geometry, then back to dictionary
    geoms = [shape(feat['geometry']) for feat in results]
    return [mapping(g) for g in geoms]

def save_geojson(polygons, output_path, crs):
    """Saves the vector data to a .geojson file."""
    fc = {
        "type": "FeatureCollection",
        "crs": { "type": "name", "properties": { "name": str(crs) } },
        "features": [
            {"type": "Feature", "properties": {"id": i}, "geometry": poly}
            for i, poly in enumerate(polygons)
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(fc, f)
    print(f"Saved output to: {output_path}")

# --- Main CLI Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Coastline Extraction Tool")
    parser.add_argument("input", help="Path to input .tif satellite image")
    parser.add_argument("output", help="Path to save output .geojson file")
    
    args = parser.parse_args()
    
    try:
        print(f"Starting processing for: {args.input}")
        
        # 1. Load
        img_data = load_image(args.input)
        
        # 2. Process
        water_mask = calculate_water_mask(img_data)
        
        # 3. Vectorize
        vectors = vectorize_mask(water_mask, img_data['transform'])
        
        # 4. Save
        save_geojson(vectors, args.output, img_data['crs'])
        
        print("Pipeline Complete.")
        
    except Exception as e:
        print(f"Error: {e}")