from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt

def create_dummy_island():
    """
    Creates a simple square 'island' to practice geospatial coordinates.
    """
    island_coords = [(0, 0), (0, 10), (10, 10), (10, 0)]
    island_poly = Polygon(island_coords)
    
    coastline = island_poly.boundary
    
    print(f"Island Area: {island_poly.area} sq units")
    print(f"Coastline Length: {coastline.length} units")
    
    return island_poly, coastline

if __name__ == "__main__":
    island, coast = create_dummy_island()
    print("Success! Geospatial libraries are working.")