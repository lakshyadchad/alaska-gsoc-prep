import rasterio
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np

image_path = "F:/alaska gsoc/CoastlineExtraction/sample_data/PlanetLabs/20160909_213103_0e19_3B_AnalyticMS_SR_clip.tif"

def apply_otsu_threshold():
    with rasterio.open(image_path) as src:
        print(f"Processing: {src.name}")
        
        # 1. Read Data
        green = src.read(2)
        nir = src.read(4)
        valid_mask = src.read_masks(1)
        
        # 2. Calculate NDWI
        with np.errstate(divide='ignore', invalid='ignore'):
            ndwi = (green.astype(float) - nir.astype(float)) / (green + nir + 0.00001)
        
        # 3. Clean Data for Statistics
        # We only want to calculate the threshold based on REAL data, not the empty padding.
        # We flatten the array to a 1D list of numbers for the algorithm.
        valid_ndwi_pixels = ndwi[valid_mask == 255]
        
        # Remove NaNs if any slipped through
        valid_ndwi_pixels = valid_ndwi_pixels[~np.isnan(valid_ndwi_pixels)]

        # 4. Calculate Otsu Threshold
        # This is the "Smart" step. The computer decides the cutoff.
        otsu_thresh = threshold_otsu(valid_ndwi_pixels)
        print(f"Computer calculated optimal threshold: {otsu_thresh:.4f}")
        
        # 5. Apply Both Thresholds for Comparison
        mask_fixed = (ndwi > 0.1) & (valid_mask == 255)
        mask_otsu = (ndwi > otsu_thresh) & (valid_mask == 255)
        
        # --- Visualization ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot A: The Histogram (How the computer made the decision)
        ax_hist = axes[0, 0]
        ax_hist.hist(valid_ndwi_pixels, bins=256, range=(-1, 1), color='gray', alpha=0.7)
        ax_hist.axvline(otsu_thresh, color='red', linestyle='--', linewidth=2, label=f'Otsu ({otsu_thresh:.2f})')
        ax_hist.axvline(0.1, color='blue', linestyle='--', linewidth=2, label='Fixed (0.1)')
        ax_hist.set_title("NDWI Histogram & Thresholds")
        ax_hist.legend()
        
        # Plot B: The Raw Image (NIR Band)
        axes[0, 1].imshow(nir, cmap='gray')
        axes[0, 1].set_title("Original NIR Image")
        
        # Plot C: Fixed Threshold Result
        axes[1, 0].imshow(mask_fixed, cmap='gray')
        axes[1, 0].set_title("Method A: Fixed Threshold (0.1)")
        
        # Plot D: Otsu Threshold Result
        axes[1, 1].imshow(mask_otsu, cmap='gray')
        axes[1, 1].set_title(f"Method B: Otsu Threshold ({otsu_thresh:.2f})")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    apply_otsu_threshold()