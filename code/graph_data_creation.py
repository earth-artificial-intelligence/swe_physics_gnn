import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
from torch_geometric.utils import degree

# 📍 File path
file_path = '/media/volume1/all_points_final_merged_training.csv'
chunksize = 500000

# 📋 Intended columns
base_columns = [
    'date', 'lat', 'lon', 'Elevation', 'Slope', 'Aspect', 'Curvature', 'Northness', 'Eastness',
    'SWE', 'fsca', 'air_temperature_tmmx', 'air_temperature_tmmn',
    'potential_evapotranspiration', 'relative_humidity_rmax', 'relative_humidity_rmin',
    'mean_vapor_pressure_deficit', 'wind_speed', 'snodas_mask', 'water_year', 'swe_value'
]

time_series_cols = [
    f"{col}_{i}" for col in [
        "SWE", "air_temperature_tmmx", "air_temperature_tmmn", "potential_evapotranspiration",
        "relative_humidity_rmax", "relative_humidity_rmin", "mean_vapor_pressure_deficit",
        "precipitation_amount", "wind_speed", "fsca"
    ] for i in range(1, 8)
]

cumulative_cols = [
    "cumulative_SWE", "cumulative_air_temperature_tmmn", "cumulative_air_temperature_tmmx",
    "cumulative_potential_evapotranspiration", "cumulative_mean_vapor_pressure_deficit",
    "cumulative_precipitation_amount", "cumulative_relative_humidity_rmax",
    "cumulative_relative_humidity_rmin", "cumulative_wind_speed", "cumulative_fsca"
]

intended_columns = base_columns + time_series_cols + cumulative_cols

# 🔍 Check available columns
print("🔍 Checking available columns in CSV...")
preview = pd.read_csv(file_path, nrows=1)
available_columns = set(preview.columns)
useful_columns = [col for col in intended_columns if col in available_columns]
if 'swe_value' not in useful_columns:
    useful_columns.append('swe_value')

missing_columns = [col for col in intended_columns if col not in available_columns]
if missing_columns:
    print(f"⚠️ Missing columns skipped: {missing_columns}")

# 📥 Load dataset
print("\n📥 Loading dataset...")
df_list = []
for chunk in pd.read_csv(file_path, usecols=useful_columns, chunksize=chunksize):
    if 'date' in chunk.columns:
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        chunk['day_of_year'] = chunk['date'].dt.dayofyear
    else:
        chunk['day_of_year'] = 1

    if 'swe_value' in chunk.columns:
        chunk = chunk[(chunk['swe_value'] >= 0) & (chunk['swe_value'] < 3000)]

    df_list.append(chunk)

data = pd.concat(df_list, ignore_index=True)
data = data.drop_duplicates()
print(f"✅ Loaded shape: {data.shape}")

# 🧮 Binning
grid_size = 0.5
num_time_bins = 8
data['lat_bin'] = (data['lat'] // grid_size).astype(int)
data['lon_bin'] = (data['lon'] // grid_size).astype(int)
data['dayofyear_bin'] = (data['day_of_year'] // (365 / num_time_bins)).astype(int)
data['grid_id'] = data['lat_bin'].astype(str) + "_" + data['lon_bin'].astype(str) + "_" + data['dayofyear_bin'].astype(str)

# 📦 Aggregation
agg_cols = {col: 'mean' for col in useful_columns if col not in ['date', 'lat', 'lon']}
agg_cols.update({'lat': 'mean', 'lon': 'mean', 'day_of_year': 'mean'})
if 'date' in useful_columns:
    agg_cols['date'] = 'first'
agg_cols['swe_value'] = 'mean'

merged_nodes = data.groupby('grid_id').agg(agg_cols).reset_index()

# ⏳ Temporal encoding
merged_nodes['sin_doy'] = np.sin(2 * np.pi * merged_nodes['day_of_year'] / 365)
merged_nodes['cos_doy'] = np.cos(2 * np.pi * merged_nodes['day_of_year'] / 365)

# 🌍 Spatial encoding
merged_nodes['lat_rad'] = np.radians(merged_nodes['lat'])
merged_nodes['lon_rad'] = np.radians(merged_nodes['lon'])
merged_nodes['sin_lat'] = np.sin(merged_nodes['lat_rad'])
merged_nodes['cos_lat'] = np.cos(merged_nodes['lat_rad'])
merged_nodes['sin_lon'] = np.sin(merged_nodes['lon_rad'])
merged_nodes['cos_lon'] = np.cos(merged_nodes['lon_rad'])

# 🧪 Final feature selection
exclude = ['grid_id', 'lat', 'lon', 'lat_rad', 'lon_rad', 'date', 'day_of_year', 'swe_value']
feature_columns = [col for col in merged_nodes.columns if col not in exclude]
feature_columns += ['sin_lat', 'cos_lat', 'sin_lon', 'cos_lon']

# ⚙️ Normalize & filter correlations
scaler = StandardScaler()
node_features_scaled = scaler.fit_transform(merged_nodes[feature_columns])
df_scaled = pd.DataFrame(node_features_scaled, columns=feature_columns)

corr = df_scaled.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
print(f"🧹 Dropping highly correlated features: {to_drop}")
df_scaled.drop(columns=to_drop, inplace=True)

# Convert to tensor
node_features = torch.tensor(df_scaled.values, dtype=torch.float)
labels = torch.tensor(merged_nodes['swe_value'].values, dtype=torch.float)

# 🌐 Build edges with KDTree
coordinates = merged_nodes[['lat', 'lon']].values
tree = KDTree(coordinates)
k = 6
threshold = 0.5
distances, neighbors = tree.query(coordinates, k=k+1)

edge_list = []
for i in range(len(neighbors)):
    for j in range(1, k+1):
        if distances[i][j] < threshold:
            edge_list.append((i, neighbors[i][j]))

edges = np.array(edge_list).T
edge_index = torch.tensor(edges, dtype=torch.long) if edges.size > 0 else torch.empty((2, 0), dtype=torch.long)

# 🧵 Create Graph
graph_data = Data(x=node_features, edge_index=edge_index, y=labels)
graph_data.dates = merged_nodes['date'].astype(str).tolist() if 'date' in merged_nodes.columns else []

# 📊 Graph Summary
print("\n📊 Graph Summary")
print(f"🔹 Nodes: {graph_data.num_nodes}")
print(f"🔹 Edges: {graph_data.num_edges}")
print(f"🔹 Features per node: {graph_data.num_node_features}")
deg = degree(edge_index[0], num_nodes=graph_data.num_nodes)
print(f"   • Min Degree : {deg.min().item()}")
print(f"   • Max Degree : {deg.max().item()}")
print(f"   • Mean Degree: {deg.float().mean():.2f}")
print(f"   • Isolated   : {(deg == 0).sum().item()} nodes")

# 💾 Save graph
save_path = "/media/volume1/gnn_graph_with_time_all_states_2.pt"
torch.save(graph_data, save_path)
print(f"\n✅ Graph data saved at: {save_path}")
