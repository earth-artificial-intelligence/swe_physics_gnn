import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import numpy as np

# Load dataset
file_path = '/media/volume1/snotel_ghcnd_stations_4yrs_all_cols_log10.csv'
print("Loading dataset...")
data = pd.read_csv(file_path)
print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

# Define Bounding Box (Western US region)
LAT_MAX, LON_MAX = 48.864715, -120.190430
LAT_MIN, LON_MIN = 45.798170, -124.145508

# Step 1: Filter Data Within the Bounding Box
print("Filtering data within the bounding box...")
data = data[
    (data['lat'] <= LAT_MAX) & (data['lat'] >= LAT_MIN) & 
    (data['lon'] <= LON_MAX) & (data['lon'] >= LON_MIN)
]
print(f"Filtered dataset with {data.shape[0]} rows.")

# Step 2: Apply Grid-Based Sampling
grid_size = 0.05  # Reduced grid size to increase node density
lat_bins = np.arange(LAT_MIN, LAT_MAX, grid_size)
lon_bins = np.arange(LON_MIN, LON_MAX, grid_size)

sampled_data = []
for lat_bin in lat_bins:
    for lon_bin in lon_bins:
        subset = data[
            (data['lat'] >= lat_bin) & (data['lat'] < lat_bin + grid_size) &
            (data['lon'] >= lon_bin) & (data['lon'] < lon_bin + grid_size)
        ]
        if not subset.empty:
            sampled_data.append(subset.sample(1, random_state=42))  

sampled_data = pd.concat(sampled_data)
print(f"Grid-based sampled dataset with {sampled_data.shape[0]} rows.")

# Step 3: Normalize Features
selected_features = [
    'swe_value', 'SWE', 'fsca', 'air_temperature_tmmx', 
    'air_temperature_tmmn', 'potential_evapotranspiration', 
    'relative_humidity_rmax', 'Elevation', 'Slope', 'Curvature', 
    'Aspect', 'Eastness', 'Northness'
]

print("Normalizing features...")
scaler = StandardScaler()
sampled_data[selected_features] = scaler.fit_transform(sampled_data[selected_features])

# Step 4: Normalize Labels
print("Normalizing target variable (swe_value)...")
label_scaler = StandardScaler()
sampled_data['swe_value'] = label_scaler.fit_transform(sampled_data['swe_value'].values.reshape(-1, 1))

# Step 5: Save Normalized Dataset
output_file_path = 'normalized_grid_sampled_snotel.csv'
sampled_data.to_csv(output_file_path, index=False)
print(f"Normalized sampled dataset saved as '{output_file_path}'.")

# Step 6: Create Node Features Tensor
node_features = torch.tensor(sampled_data[selected_features].values, dtype=torch.float)
print(f"Node features tensor created with shape: {node_features.shape}")

# Step 7: Create Labels Tensor
labels = torch.tensor(sampled_data['swe_value'].values, dtype=torch.float)
print(f"Labels tensor created with shape: {labels.shape}")

# Step 8: Build KDTree and Construct Edges Using Distance Threshold
print("Building KDTree and constructing edges using distance threshold...")
coordinates = sampled_data[['lat', 'lon']].values
tree = KDTree(coordinates)

k = 30  
radius = 0.2
edges = []
for i in range(len(coordinates)):
    knn_neighbors = tree.query(coordinates[i], k=k+1)[1][1:]  
    radius_neighbors = tree.query_ball_point(coordinates[i], r=radius)  
    
    for neighbor in set(knn_neighbors).union(radius_neighbors): 
        if i != neighbor:
            edges.append((i, neighbor))


print(f"Number of edges created: {len(edges)}")

# Convert to Edge Index Tensor
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(f"Edge index tensor created with shape: {edge_index.shape}")

# Step 9: Create PyTorch Geometric Graph Data Object
graph_data = Data(x=node_features, edge_index=edge_index, y=labels)

# Save the PyTorch Geometric Data Object
graph_data_output_path = '/media/volume1/gat_spatial_training_data_sampled_radius_final_01.pt'
torch.save(graph_data, graph_data_output_path)
print(f"PyTorch Geometric Data object saved as '{graph_data_output_path}'.")