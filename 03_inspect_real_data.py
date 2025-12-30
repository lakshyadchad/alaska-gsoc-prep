import rasterio
import matplotlib.pyplot as plt
import os

# Define path to the sample data in the OTHER folder
# We use ".." to go up one level, then down into CoastlineExtraction
image_path = "F:/alaska gsoc/CoastlineExtraction/sample_data/PlanetLabs/20160909_213103_0e19_3B_AnalyticMS_SR_clip.tif"

def inspect_tif():
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Could not find file at {image_path}")
        return

    print(f"Found image! Opening...")
    
    with rasterio.open(image_path) as src:
        print(f"--- Metadata ---")
        print(f"Width: {src.width}, Height: {src.height}")
        print(f"Bands: {src.count}")
        print(f"CRS (Coordinate System): {src.crs}")
        
        # Read the first band (Blue)
        blue_band = src.read(1)
        
        # Plot it
        plt.figure(figsize=(10, 10))
        plt.imshow(blue_band, cmap='gray')
        plt.title(f"Real PlanetLabs Image - Band 1\n{image_path}")
        plt.colorbar()
        plt.show()

if __name__ == "__main__":
    inspect_tif()