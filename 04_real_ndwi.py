import rasterio
import matplotlib.pyplot as plt

# Path to the specific Deering image
image_path = "F:/alaska gsoc/CoastlineExtraction/sample_data/PlanetLabs/20160909_213103_0e19_3B_AnalyticMS_SR_clip.tif"

def process_real_ndwi():
    with rasterio.open(image_path) as src:
        print(f"Opening: {src.name}")
        
        # 1. Read the specific bands we need (Note: Python uses 0-indexing, but Rasterio uses 1-indexing)
        # PlanetLabs Order: 1=Blue, 2=Green, 3=Red, 4=NIR
        green = src.read(2).astype(float)
        nir = src.read(4).astype(float)
        
        # 2. Calculate NDWI
        # Formula: (Green - NIR) / (Green + NIR)
        # We add a tiny epsilon (0.00001) to avoid dividing by zero
        ndwi = (green - nir) / (green + nir + 0.00001)
        
        # 3. Create a Threshold Mask (The "Coastline" decision)
        # Usually, values > 0.0 or 0.1 are water.
        water_mask = ndwi > 0.1 
        
        # Visualization 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot 1: The NDWI Index (The math)
        im = ax1.imshow(ndwi, cmap='RdYlBu')
        ax1.set_title("NDWI Index\n(Blue = Water, Red = Land)")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
        # Plot 2: The Binary Mask (The result)
        ax2.imshow(water_mask, cmap='gray')
        ax2.set_title("Generated Water Mask\n(White = Water, Black = Land)")
        
        plt.tight_layout()
        plt.show()
        
        print("Analysis Complete. Verify the coastline shape in the plot.")

if __name__ == "__main__":
    process_real_ndwi()