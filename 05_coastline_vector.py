import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import matplotlib.pyplot as plt
import numpy as np

image_path = "F:/alaska gsoc/CoastlineExtraction/sample_data/PlanetLabs/20160909_213103_0e19_3B_AnalyticMS_SR_clip.tif"

def extract_vector_coastline():
    with rasterio.open(image_path) as src:
        print(f"Processing: {src.name}")
        
        # 1. Read Bands
        green = src.read(2)
        nir = src.read(4)
        
        # 2. Read the Valid Data Mask (Fixes the "White Border" bug)
        valid_data_mask = src.read_masks(1)
        
        # 3. Calculate NDWI
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green.astype(float) - nir.astype(float)) / (green + nir + 0.00001)
        
        # 4. Create Binary Water Mask
        water_mask = (ndwi > 0.1) & (valid_data_mask == 255)
        water_mask_int = water_mask.astype(np.uint8)

        # 5. Vectorize!
        print("Vectorizing coastline...")
        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(
                shapes(water_mask_int, mask=valid_data_mask==255)
            )
            if v == 1
        )

        water_polygons = [shape(feat['geometry']) for feat in results]
        
        # 6. Visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(nir, cmap='gray')
        
        # Plot the Vectors (With error handling for Complex Shapes)
        for poly in water_polygons:
            if poly.area > 500: # Filter noise
                boundary = poly.boundary
                
                # Check: Is it one line or many lines?
                if boundary.geom_type == 'MultiLineString':
                    for line in boundary.geoms:
                        x, y = line.xy
                        ax.plot(x, y, color='cyan', linewidth=2)
                elif boundary.geom_type == 'LineString':
                    x, y = boundary.xy
                    ax.plot(x, y, color='cyan', linewidth=2)

        ax.set_title("Final Output: Vectorized Coastline (Cyan Line)")
        plt.show()
        
        print(f"Found {len(water_polygons)} water features.")
        print("Success! The code now handles islands and complex shorelines.")

if __name__ == "__main__":
    extract_vector_coastline()