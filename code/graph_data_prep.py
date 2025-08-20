import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import numpy as np

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

# Step 2: Spatially aware sampling
sample_size = min(50000, len(data))  # Adjust sample size dynamically
coordinates = data[['lat', 'lon']].values

print("Building KDTree for spatial sampling...")
tree = KDTree(coordinates)
indices = [np.random.randint(0, len(data))]  # Start with a random point

while len(indices) < sample_size:
    ref_point = coordinates[indices[-1]]  # Last added point
    neighbors = tree.query_ball_point(ref_point, r=0.01)  # Get neighbors within radius
    if len(neighbors) > 1:
        new_idx = np.random.choice(neighbors)
        if new_idx not in indices:
            indices.append(new_idx)

sampled_data = data.iloc[indices]
print(f"Spatially sampled dataset with {sampled_data.shape[0]} rows.")

# Step 3: Normalize Features
selected_features = [
    'lat', 'lon', 'Elevation', 'Slope', 'SWE', 'snow_depth',
    'air_temperature_observed_f', 'precipitation_amount',
    'relative_humidity_rmin', 'relative_humidity_rmax', 'wind_speed',
    'cumulative_SWE', 'cumulative_precipitation_amount'
]

print("Normalizing features...")
scaler = StandardScaler()
sampled_data[selected_features] = scaler.fit_transform(sampled_data[selected_features])

# Step 4: Normalize Labels (swe_value)
print("Normalizing target variable (swe_value)...")
label_scaler = StandardScaler()
sampled_data['swe_value'] = label_scaler.fit_transform(sampled_data['swe_value'].values.reshape(-1, 1))

# Save normalized dataset
output_file_path = 'normalized_spatial_sampled_snotel.csv'
sampled_data.to_csv(output_file_path, index=False)
print(f"Normalized sampled dataset saved as '{output_file_path}'.")

# Step 5: Create Node Features Tensor
node_features = torch.tensor(sampled_data[selected_features].values, dtype=torch.float)
print(f"Node features tensor created with shape: {node_features.shape}")

# Step 6: Create Labels Tensor
labels = torch.tensor(sampled_data['swe_value'].values, dtype=torch.float)
print(f"Labels tensor created with shape: {labels.shape}")

# Step 7: Build KDTree and Construct Edges
print("Building KDTree and constructing edges...")
coordinates = sampled_data[['lat', 'lon']].values
tree = KDTree(coordinates)
threshold = 0.005  # Adjust as needed

batch_edges = tree.query_pairs(r=threshold)
tedges = list(batch_edges)
print(f"Number of edges created: {len(tedges)}")

# Convert to Edge Index Tensor
edge_index = torch.tensor(tedges, dtype=torch.long).t().contiguous()
print(f"Edge index tensor created with shape: {edge_index.shape}")

# Step 8: Create PyTorch Geometric Graph Data Object
graph_data = Data(x=node_features, edge_index=edge_index, y=labels)

# Save the PyTorch Geometric Data Object
graph_data_output_path = '/media/volume1/gat_spatial_training_data_sampled.pt'
torch.save(graph_data, graph_data_output_path)
print(f"PyTorch Geometric Data object saved as '{graph_data_output_path}'.")
