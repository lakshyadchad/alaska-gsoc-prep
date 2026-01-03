import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show

# 1. Define Paths
satellite_img_path = "F:/alaska gsoc/CoastlineExtraction/sample_data/PlanetLabs/20160909_213103_0e19_3B_AnalyticMS_SR_clip.tif"
my_result_path = "output_coastline.geojson"
usgs_truth_path = "F:/alaska gsoc/CoastlineExtraction/USGS_Coastlines/Deering_shorelines_2016.shp"

def validate_results():
    print("Loading data...")
    
    # Load Satellite Image (for background)
    src = rasterio.open(satellite_img_path)
    
    # Load Your Result
    my_coast = gpd.read_file(my_result_path)
    
    # Load Ground Truth (Official Science Data)
    # We filter for 2016 data to match our image year
    usgs = gpd.read_file(usgs_truth_path)
    if usgs.crs != my_coast.crs:
        print(f"Re-projecting USGS data from {usgs.crs} to {my_coast.crs}...")
        usgs = usgs.to_crs(my_coast.crs)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Plot Satellite Image (Band 4 - NIR for contrast)
    show((src, 4), ax=ax, cmap='gray', alpha=0.6)
    
    # 2. Plot USGS Ground Truth (RED DASHED LINE)
    # This is what the scientists drew by hand.
    usgs.plot(ax=ax, color='red', linestyle='--', linewidth=2, label='USGS Ground Truth (Manual)')
    
    # 3. Plot YOUR Automated Result (CYAN SOLID LINE)
    my_coast.plot(ax=ax, color='cyan', linewidth=2, alpha=0.8, label='My Automated Pipeline')
    
    # Legend & styling
    # We create custom legend handles because .plot() sometimes messes up legends in Geopandas
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', linestyle='--', lw=2),
                    Line2D([0], [0], color='cyan', lw=2)]
    ax.legend(custom_lines, ['USGS Manual (Ground Truth)', 'My Automated Extraction (LCC)'])
    
    plt.title("Validation: Automated Extraction vs. Manual Digitization")
    plt.show()
    
    print("Validation Plot Generated.")
    print("Check: How closely does the Cyan line follow the Red line?")

if __name__ == "__main__":
    validate_results()