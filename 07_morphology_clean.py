import rasterio
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import numpy as np

image_path = "F:/alaska gsoc/CoastlineExtraction/sample_data/PlanetLabs/20160909_213103_0e19_3B_AnalyticMS_SR_clip.tif"

def keep_only_largest():
    with rasterio.open(image_path) as src:
        print(f"Processing: {src.name}")
        
        # 1. Setup Data
        green = src.read(2)
        nir = src.read(4)
        valid_mask = src.read_masks(1)
        
        # 2. Get the Noisy Mask (Otsu)
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green.astype(float) - nir.astype(float)) / (green + nir + 0.00001)
        
        valid_pixels = ndwi[valid_mask == 255]
        valid_pixels = valid_pixels[~np.isnan(valid_pixels)]
        otsu_thresh = threshold_otsu(valid_pixels)
        mask_noisy = (ndwi > otsu_thresh) & (valid_mask == 255)
        
        # 3. The "Keep Largest" Logic
        # Label every distinct island with a number (1, 2, 3...)
        labeled_mask = label(mask_noisy)
        
        # Get properties of every island
        regions = regionprops(labeled_mask)
        
        if not regions:
            print("❌ No water found!")
            return

        # Find the one with the maximum area
        largest_region = max(regions, key=lambda r: r.area)
        print(f"🌊 Largest Region Area: {largest_region.area} pixels")
        
        # Create a new blank mask and paint ONLY the largest region
        mask_clean = np.zeros_like(mask_noisy)
        mask_clean[labeled_mask == largest_region.label] = 1
        
        # Calculate how much junk we removed
        noise_area_removed = np.sum(mask_noisy) - np.sum(mask_clean)
        print(f"🗑️ Deleted {noise_area_removed} pixels of noise.")

        # --- Visualization ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Plot 1: The messy one
        ax1.imshow(mask_noisy, cmap='gray')
        ax1.set_title("Before: Otsu + Noise")
        
        # Plot 2: The clean one
        ax2.imshow(mask_clean, cmap='gray')
        ax2.set_title(f"After: Kept Largest Component Only\n(Area: {largest_region.area})")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    keep_only_largest()