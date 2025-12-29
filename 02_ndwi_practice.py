import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin

def generate_dummy_satellite_image():
    """
    Creates a fake 2-band satellite image (Band 1: Green, Band 2: NIR).
    We will simulate a 'river' running through a 'forest'.
    """
    height, width = 100, 100
    
    # 1. Create empty bands
    green_band = np.ones((height, width), dtype=np.float32) * 0.2  # Low reflectance
    nir_band = np.ones((height, width), dtype=np.float32) * 0.6    # High reflectance (Vegetation)
    
    # 2. Draw a "River" (Water has HIGH Green, LOW NIR relative to vegetation)
    # Let's make a vertical river in the middle
    green_band[:, 40:60] = 0.4  # Water reflects some green
    nir_band[:, 40:60] = 0.05   # Water absorbs NIR (Very dark)

    return green_band, nir_band

def calculate_ndwi(green, nir):
    """
    Formula: (Green - NIR) / (Green + NIR)
    Values > 0.3 are usually water.
    Values <= 0.3 are usually land/vegetation.
    """
    # Avoid division by zero
    denominator = green + nir
    denominator[denominator == 0] = 0.0001
    
    ndwi = (green - nir) / denominator
    return ndwi

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Get Data
    green, nir = generate_dummy_satellite_image()
    
    # 2. Process
    ndwi = calculate_ndwi(green, nir)
    
    # 3. Visualise
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(green, cmap='Greens')
    ax1.set_title("Green Band (Visible)")
    
    ax2.imshow(nir, cmap='Reds')
    ax2.set_title("NIR Band (Infrared)")
    
    # The Magic: NDWI makes the river bright yellow/green and land dark
    im = ax3.imshow(ndwi, cmap='RdYlBu') 
    ax3.set_title("NDWI Calculation")
    plt.colorbar(im, ax=ax3)
    
    plt.show()
    
    print(f"Max NDWI value (River): {np.max(ndwi):.2f}")
    print(f"Min NDWI value (Land): {np.min(ndwi):.2f}")