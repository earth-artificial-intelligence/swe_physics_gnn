import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from torch_geometric.utils import degree, to_undirected, add_self_loops

# Load Data
file_path = '/media/volume1/snotel_ghcnd_stations_4yrs_all_cols_log10.csv'
chunksize = 500000
#  latitude & longitude to cover Washington, Oregon, and Idaho
min_lat, max_lat = 42.365162, 48.981824
min_lon, max_lon = -123.842534, -111.627646


print("Loading dataset...")

useful_columns = [
    'SWE', 'swe_value', 'relative_humidity_rmin',
    'potential_evapotranspiration', 'air_temperature_tmmx',
    'relative_humidity_rmax', 'mean_vapor_pressure_deficit',
    'air_temperature_tmmn', 'wind_speed', 'Elevation', 'Aspect',
    'Curvature', 'Northness', 'Eastness', 'fsca', 'Slope', 'SWE_1',
    'air_temperature_tmmn_1', 'potential_evapotranspiration_1',
    'mean_vapor_pressure_deficit_1', 'relative_humidity_rmax_1',
    'relative_humidity_rmin_1', 'air_temperature_tmmx_1', 'wind_speed_1',
    'fsca_1', 'SWE_2', 'air_temperature_tmmn_2',
    'potential_evapotranspiration_2', 'mean_vapor_pressure_deficit_2',
    'relative_humidity_rmax_2', 'relative_humidity_rmin_2',
    'air_temperature_tmmx_2', 'wind_speed_2', 'fsca_2', 'SWE_3',
    'air_temperature_tmmn_3', 'potential_evapotranspiration_3',
    'mean_vapor_pressure_deficit_3', 'relative_humidity_rmax_3',
    'relative_humidity_rmin_3', 'air_temperature_tmmx_3', 'wind_speed_3',
    'fsca_3', 'SWE_4', 'air_temperature_tmmn_4',
    'potential_evapotranspiration_4', 'mean_vapor_pressure_deficit_4',
    'relative_humidity_rmax_4', 'relative_humidity_rmin_4',
    'air_temperature_tmmx_4', 'wind_speed_4', 'fsca_4', 'SWE_5',
    'air_temperature_tmmn_5', 'potential_evapotranspiration_5',
    'mean_vapor_pressure_deficit_5', 'relative_humidity_rmax_5',
    'relative_humidity_rmin_5', 'air_temperature_tmmx_5', 'wind_speed_5',
    'fsca_5', 'SWE_6', 'air_temperature_tmmn_6',
    'potential_evapotranspiration_6', 'mean_vapor_pressure_deficit_6',
    'relative_humidity_rmax_6', 'relative_humidity_rmin_6',
    'air_temperature_tmmx_6', 'wind_speed_6', 'fsca_6', 'SWE_7',
    'air_temperature_tmmn_7', 'potential_evapotranspiration_7',
    'mean_vapor_pressure_deficit_7', 'relative_humidity_rmax_7',
    'relative_humidity_rmin_7', 'air_temperature_tmmx_7', 'wind_speed_7',
    'fsca_7', 'water_year', 'snodas_mask'
]

df_list = []
total_rows = 0
for chunk in pd.read_csv(file_path, usecols=useful_columns, parse_dates=['date'], chunksize=chunksize):
    chunk['dayofyear'] = chunk['date'].dt.dayofyear  
    total_rows += len(chunk)
    filtered_chunk = chunk[
        (chunk["lat"] >= min_lat) & (chunk["lat"] <= max_lat) &
        (chunk["lon"] >= min_lon) & (chunk["lon"] <= max_lon)
    ]
    df_list.append(filtered_chunk)

data = pd.concat(df_list, ignore_index=True)
print(f"Total rows in raw dataset: {total_rows}")
print(f"Rows remaining after lat/lon filtering: {data.shape[0]}")

# Check for missing values
missing_values = data[['date', 'lat', 'lon', 'swe_value']].isnull().sum()
print(f"Missing values count per column:\n{missing_values}")

# Remove duplicates
num_duplicates = data.duplicated().sum()
print(f"Found {num_duplicates} duplicate rows.")
data = data.drop_duplicates()
print(f"Dataset size after deduplication: {data.shape[0]}")

# Merge records for the same station on the same day
print("Merging records for the same station on the same day...")
merge_columns = [
    'Elevation', 'Slope', 'SWE', 'snow_depth',
    'air_temperature_observed_f', 'precipitation_amount',
    'relative_humidity_rmin', 'relative_humidity_rmax', 'wind_speed',
    'cumulative_SWE', 'cumulative_precipitation_amount', 'swe_value'
]
data = data.groupby(['lat', 'lon', 'date'], as_index=False)[merge_columns].mean()
data['dayofyear'] = pd.to_datetime(data['date']).dt.dayofyear
print(f" Merged dataset size: {data.shape[0]}")

# Round numerical values
print(" Rounding numerical values to 3 decimals...")
numerical_cols = list(set(merge_columns) & set(data.columns))  
data[numerical_cols] = data[numerical_cols].round(3)

# Merging nodes to reduce graph size
grid_size = 0.01  
time_bins = 12  #  Increased from 10 to 12 bins

data['lat_bin'] = (data['lat'] // grid_size).astype(int)
data['lon_bin'] = (data['lon'] // grid_size).astype(int)
data['time_bin'] = (data['dayofyear'] // (365 // time_bins)).astype(int)
data['grid_id'] = data['lat_bin'].astype(str) + "_" + data['lon_bin'].astype(str) + "_" + data['time_bin'].astype(str)

merged_nodes = data.groupby('grid_id').agg({
    'Elevation': 'mean', 'Slope': 'mean', 'SWE': 'mean', 'snow_depth': 'mean',
    'air_temperature_observed_f': 'mean', 'precipitation_amount': 'mean',
    'relative_humidity_rmin': 'mean', 'relative_humidity_rmax': 'mean', 'wind_speed': 'mean',
    'cumulative_SWE': 'mean', 'cumulative_precipitation_amount': 'mean', 'swe_value': 'mean',
    'lat': 'mean', 'lon': 'mean', 'dayofyear': 'mean'
}).reset_index()

print(f" Merged nodes: {merged_nodes.shape[0]} from {data.shape[0]} original records.")

# Normalizing features
feature_columns = [
    'Elevation', 'Slope', 'SWE', 'snow_depth',
    'air_temperature_observed_f', 'precipitation_amount',
    'relative_humidity_rmin', 'relative_humidity_rmax', 'wind_speed',
    'cumulative_SWE', 'cumulative_precipitation_amount'
]

merged_nodes[feature_columns] = merged_nodes[feature_columns].fillna(merged_nodes[feature_columns].median())

scaler = StandardScaler()
merged_nodes[feature_columns] = scaler.fit_transform(merged_nodes[feature_columns].values)

node_features = torch.tensor(merged_nodes[feature_columns].values, dtype=torch.float)
labels = torch.tensor(merged_nodes['swe_value'].values, dtype=torch.float)

print(f" Normalization completed. Node feature shape: {node_features.shape}")

# Adaptive spatial threshold based on data distribution
coordinates = merged_nodes[['lat', 'lon', 'dayofyear']].values  
tree = KDTree(coordinates)

avg_degree = max(5, min(10, int(len(merged_nodes) ** 0.5 / 10)))
distances, neighbors = tree.query(coordinates, k=avg_degree + 1)

spatial_threshold = np.percentile(distances[:, 1:], 75)  #  Adaptive threshold
time_weight = 0.07  

def time_aware_distance(p1, p2):
    spatial_dist = np.linalg.norm(p1[:2] - p2[:2])  
    time_dist = abs(p1[2] - p2[2]) * time_weight  
    return spatial_dist + time_dist

print("Rebuilding KDTree and constructing edges...")

edge_list = []
for i in range(len(neighbors)):
    for j in range(1, avg_degree + 1):
        if time_aware_distance(coordinates[i], coordinates[neighbors[i][j]]) < spatial_threshold:
            edge_list.append((i, neighbors[i][j]))

edges = np.array(edge_list).T
edge_index = torch.tensor(edges, dtype=torch.long) if edges.shape[1] > 0 else torch.empty((2, 0), dtype=torch.long)

# Ensure bidirectional edges
edge_index = to_undirected(edge_index)

# Graph Pruning: Remove low-degree nodes
deg = degree(edge_index[0], num_nodes=merged_nodes.shape[0])
valid_nodes = deg > 1  
filtered_node_mask = valid_nodes.nonzero(as_tuple=True)[0]

# Filter node features & labels
node_features = node_features[filtered_node_mask]
labels = labels[filtered_node_mask]

# Reconstruct Edge Index after filtering
node_map = {old_idx: new_idx for new_idx, old_idx in enumerate(filtered_node_mask.tolist())}
new_edges = [(node_map[i], node_map[j]) for i, j in edge_index.T.tolist() if i in node_map and j in node_map]
edge_index = torch.tensor(new_edges, dtype=torch.long).T if len(new_edges) > 0 else torch.empty((2, 0), dtype=torch.long)

print(f"Final Graph: Nodes = {len(filtered_node_mask)}, Edges = {edge_index.shape[1]}")

# Save graph
graph_data = Data(x=node_features, edge_index=edge_index, y=labels)
torch.save(graph_data, "/media/volume1/gnn_3states.pt")

print(" Graph successfully saved.")
