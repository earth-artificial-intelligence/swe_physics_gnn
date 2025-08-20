import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import pandas as pd
import os
from datetime import datetime

file_path = '/media/volume1/snotel_ghcnd_stations_4yrs_all_cols_log10.csv'

print("Loading dataset...")
data = pd.read_csv(file_path)
print(f"Dataset loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

# Step 1: Filter data for the western USA region
lat_min, lat_max = 32.0, 49.0  # Covers western states
lon_min, lon_max = -125.0, -102.0  # Covers western states
filtered_data = data[(data['lat'] >= lat_min) & (data['lat'] <= lat_max) &
                     (data['lon'] >= lon_min) & (data['lon'] <= lon_max)]
print(f"Filtered dataset to western USA region with {filtered_data.shape[0]} rows.")

# Step 2: Sample fewer data points
sample_size = 50000  
sampled_data = filtered_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
print(f"Sampled dataset with {sampled_data.shape[0]} rows.")

# Step 3: Normalize features
selected_features = [
    'lat', 'lon', 'Elevation', 'Slope', 'SWE', 'snow_depth',
    'air_temperature_observed_f', 'precipitation_amount',
    'relative_humidity_rmin', 'relative_humidity_rmax', 'wind_speed',
    'cumulative_SWE', 'cumulative_precipitation_amount'
]

print("Normalizing features...")
scaler = StandardScaler()
sampled_data[selected_features] = scaler.fit_transform(sampled_data[selected_features])

output_file_path = 'normalized_sampled_snotel_west.csv'
sampled_data.to_csv(output_file_path, index=False)
print("Normalized sampled dataset saved as 'normalized_sampled_snotel_west.csv'.")

# Step 4: Create node features tensor
node_features = torch.tensor(sampled_data[selected_features].values, dtype=torch.float)
print(f"Node features tensor created with shape: {node_features.shape}")

# Step 5: Create labels tensor
labels = torch.tensor(sampled_data['swe_value'].values, dtype=torch.float)
print(f"Labels tensor created with shape: {labels.shape}")

# Step 6: Build KDTree and construct fewer edges
coordinates = sampled_data[['lat', 'lon']].values
threshold = 0.05  # Adjusted threshold to limit edges

print("Building KDTree and constructing edges...")
tree = KDTree(coordinates)
tedges = list(tree.query_pairs(r=threshold))

# Ensure bidirectional edges
edges = np.array(tedges)
edges = np.vstack((edges, edges[:, ::-1]))  # Add reversed edges
edge_index = torch.tensor(edges.T, dtype=torch.long)
print(f"Edge index tensor created with shape: {edge_index.shape}")

# Step 7: Create PyTorch Geometric graph data object
graph_data = Data(x=node_features, edge_index=edge_index, y=labels)

# Save the PyTorch Geometric Data object
graph_data_output_path = '/media/volume1/gat_training_data_west.pt'
torch.save(graph_data, graph_data_output_path)
print(f"PyTorch Geometric Data object saved as '{graph_data_output_path}'.")
