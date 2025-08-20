# SWE Physics GNN - Developer Guide

## Project Overview

This project implements Graph Neural Networks (GNNs) for Snow Water Equivalent (SWE) prediction using physical and environmental data. The codebase combines spatial data processing, graph creation, and deep learning techniques to model snow water equivalent across geographic regions.

## Data Columns and Target Variable

### Key Data Columns

#### Base Features
- **Geospatial Features**: `lat`, `lon`, `Elevation`, `Slope`, `Aspect`, `Curvature`, `Northness`, `Eastness`
- **Meteorological Features**: `air_temperature_tmmx`, `air_temperature_tmmn`, `potential_evapotranspiration`, `relative_humidity_rmax`, `relative_humidity_rmin`, `mean_vapor_pressure_deficit`, `wind_speed`
- **Snow Features**: `SWE`, `fsca` (fractional snow covered area)
- **Temporal Features**: `date`, `water_year`, `day_of_year`

#### Time Series Features
The model uses historical time series data for the following variables (with suffix _1 through _7 indicating previous days):
- `SWE_1` through `SWE_7`
- `air_temperature_tmmx_1` through `air_temperature_tmmx_7`
- `air_temperature_tmmn_1` through `air_temperature_tmmn_7`
- `potential_evapotranspiration_1` through `potential_evapotranspiration_7`
- `relative_humidity_rmax_1` through `relative_humidity_rmax_7`
- `relative_humidity_rmin_1` through `relative_humidity_rmin_7`
- `mean_vapor_pressure_deficit_1` through `mean_vapor_pressure_deficit_7`
- `precipitation_amount_1` through `precipitation_amount_7`
- `wind_speed_1` through `wind_speed_7`
- `fsca_1` through `fsca_7`

#### Cumulative Features
Accumulated values over time for key variables:
- `cumulative_SWE`
- `cumulative_air_temperature_tmmn`
- `cumulative_air_temperature_tmmx`
- `cumulative_potential_evapotranspiration`
- `cumulative_mean_vapor_pressure_deficit`
- `cumulative_precipitation_amount`
- `cumulative_relative_humidity_rmax`
- `cumulative_relative_humidity_rmin`
- `cumulative_wind_speed`
- `cumulative_fsca`

#### Derived Features
- **Temporal Encoding**: `sin_doy`, `cos_doy` (sinusoidal encoding of day of year)
- **Spatial Encoding**: `sin_lat`, `cos_lat`, `sin_lon`, `cos_lon` (spherical encoding of coordinates)

### Target Variable

The target variable for prediction is `swe_value`, which represents the Snow Water Equivalent measurement in millimeters. This is the amount of water contained within the snowpack, a critical variable for water resource management and flood forecasting.

## Environment Setup

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository
2. Run the installation script to install required dependencies:

```bash
./code/install_modules.sh
```

This will install the necessary packages including:
- pandas
- torch
- torch-geometric
- scikit-learn
- scipy
- numpy

## Data Preparation

### Data Fetching

Use the `fetch_data.sh` script to download the required datasets:

```bash
./code/fetch_data.sh
```

This will download the SNOTEL and GHCND station data to the `/media/volume1/` directory.

### Data Processing Pipeline for SWE Forecasting

1. **Data Loading and Filtering**
   - Load data from CSV files containing meteorological and snow measurements
   - Filter by geographic region (typically Western US mountain ranges)
   - Remove outliers (e.g., SWE values > 3000mm are filtered out)

2. **Feature Engineering**
   - **Spatial Features**: Latitude, longitude, elevation, slope, aspect
   - **Temporal Features**: Day of year (converted to sin/cos encoding)
   - **Meteorological Features**: Temperature, precipitation, humidity, wind speed
   - **Snow Features**: Snow depth, snow density, SWE historical values
   - **Derived Features**: Temperature gradients, cumulative precipitation

3. **Spatial Processing**
   - Grid-based binning (0.01° to 0.5° resolution)
   - KDTree-based neighbor finding
   - Edge creation based on spatial proximity and/or feature similarity

4. **Temporal Processing**
   - Time-series aggregation (daily, weekly, monthly)
   - Lag features (previous day, week values)
   - Cumulative features (e.g., accumulated precipitation)

5. **Feature Normalization**
   - StandardScaler or MinMaxScaler depending on the model
   - Target variable (SWE) is typically normalized to [0,1] range

6. **Graph Construction**
   - Node features: Combined spatial, temporal, and meteorological features
   - Edge creation: Based on spatial proximity and temporal relationships
   - Self-loops addition for better message passing

7. **Train/Test Splitting**
   - Typically 80/20 or 70/30 split
   - Optional validation set (10-20%)
   - Consideration for temporal coherence in splits

## Graph Creation

The project offers multiple approaches to graph creation, each with specific parameters and techniques:

### Basic Graph Creation (`graph_data_prep.py`)

Creates a simple graph with spatial sampling and distance-based edge creation:

```bash
python code/graph_data_prep.py
```

#### Key Parameters
- **Bounding Box**: Western US region (LAT_MIN=45.798170, LAT_MAX=48.864715, LON_MIN=-124.145508, LON_MAX=-120.190430)
- **Sample Size**: 50,000 points (maximum)
- **Edge Creation**: KDTree with radius=0.01
- **Features**: 13 selected features including elevation, slope, temperature, etc.

#### Graph Construction Process
1. Filter data within geographic bounding box
2. Apply spatial sampling using KDTree
3. Normalize features using StandardScaler
4. Create edges between points within specified radius
5. Generate PyTorch Geometric Data object

### Enhanced Graph Creation (`graph_creation.py`)

Implements more sophisticated graph construction with adaptive spatial thresholds and graph pruning:

```bash
python code/graph_creation.py
```

#### Key Parameters
- **Grid Size**: 0.01 (for spatial binning)
- **Time Bins**: 12 (for temporal discretization)
- **Adaptive Spatial Threshold**: 75th percentile of distances
- **Time Weight**: 0.07 (for time-aware distance calculation)
- **Average Degree**: Between 5 and 10, scaled with sqrt(n)/10

#### Graph Construction Process
1. Load and filter data by geographic region
2. Merge records for the same station on the same day
3. Apply grid-based binning (lat/lon/time)
4. Normalize features
5. Build KDTree with time-aware distance metric
6. Create edges using adaptive threshold
7. Ensure bidirectional edges
8. Prune low-degree nodes (degree > 1)

### Temporal Graph Creation (`graph_data_creation.py`)

Builds graphs with temporal features and time-series data:

```bash
python code/graph_data_creation.py
```

#### Key Parameters
- **Grid Size**: 0.5 (for spatial binning)
- **Time Bins**: 8 (for temporal discretization)
- **Edge Creation**: KDTree with k=6 neighbors and threshold=0.5
- **Features**: Extensive set including base, time-series, and cumulative features

#### Graph Construction Process
1. Load data in chunks to handle large datasets
2. Apply spatial and temporal binning
3. Aggregate data by grid cell
4. Add temporal encoding (sin/cos of day of year)
5. Add spatial encoding (spherical coordinates)
6. Filter highly correlated features (>0.95)
7. Create edges using KDTree with k-nearest neighbors

### State-Specific Data Creation (`state_data_creation.py`)

Creates graph data for specific US states with grid-based sampling:

```bash
python code/state_data_creation.py
```

#### Key Parameters
- **Grid Size**: 0.05 (for spatial binning)
- **Edge Creation**: Combined KNN (k=30) and radius-based (radius=0.2) approach
- **Features**: Focus on state-specific meteorological and topographical features

#### Graph Construction Process
1. Filter data for specific state boundaries
2. Apply grid-based sampling
3. Normalize features
4. Create edges using hybrid approach (KNN + radius)
5. Generate PyTorch Geometric Data object

## Model Implementation

The project includes several model implementations:

### GraphSAGE Models

#### Basic GraphSAGE (`graph_model.py`)

Implements a GraphSAGE model with multiple convolutional layers and batch normalization:

```bash
python code/graph_model.py
```

##### Model Architecture
```python
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.conv4 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.conv5 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.bn4 = BatchNorm(hidden_dim)
        self.bn5 = BatchNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout)
```

##### Key Parameters
- **input_dim**: Number of input features (varies based on dataset)
- **hidden_dim**: 512 (default)
- **output_dim**: 1 (SWE prediction)
- **dropout**: 0.3-0.4
- **learning_rate**: 1e-4
- **weight_decay**: 1e-3
- **optimizer**: Adam
- **loss_function**: SmoothL1Loss (beta=1.0)
- **scheduler**: ReduceLROnPlateau (factor=0.8, patience=50)

##### Key Features
- 5 GraphSAGE convolutional layers
- Mean aggregation
- Batch normalization after each convolutional layer
- Dropout regularization
- LeakyReLU activation
- Gradient clipping (max_norm=5.0)

#### Enhanced GraphSAGE (`graph_model_sage.py`)

Implements an improved GraphSAGE model with residual connections:

```bash
python code/graph_model_sage.py
```

##### Model Architecture
```python
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        
        # Linear layer to project input features to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim) 
        
        self.conv1 = SAGEConv(hidden_dim, hidden_dim, aggr="max")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="max")  
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr="max")

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.1)
```

##### Key Parameters
- **input_dim**: Number of input features (varies based on dataset)
- **hidden_dim**: 256 (default)
- **output_dim**: 1 (SWE prediction)
- **learning_rate**: 0.001
- **weight_decay**: 1e-4
- **optimizer**: AdamW
- **loss_function**: SmoothL1Loss (beta=1.0)
- **scheduler**: CosineAnnealingLR (T_max=300, eta_min=1e-6)

##### Key Features
- Max aggregation (instead of mean)
- Residual connections
- Input projection layer
- Adaptive dropout (0.2 for early layers, 0.1 for final layers)
- Improved gradient clipping (max_norm=3.0)
- Cosine annealing learning rate scheduler

### Physics-Informed Neural Network (`temporal_graph.py`)

Implements a Physics-Informed Neural Network (PINN) with custom physics-based loss functions:

```bash
python code/temporal_graph.py
```

#### Model Architecture
```python
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze()
```

#### Physics-Based Loss Components

```python
def pinn_physics_loss(pred, y_true, sca_pred, sca_ref, swe_model_by_elev, swe_climo_by_elev, λ1=1.0, λ2=1.0):
    model_loss = F.mse_loss(pred, y_true)                          # Data-driven loss
    sca_loss = F.l1_loss(sca_pred, sca_ref)                        # Snow Covered Area constraint
    hypsometric_loss = F.l1_loss(swe_model_by_elev, swe_climo_by_elev)  # Elevation distribution constraint
    return model_loss + λ1 * sca_loss + λ2 * hypsometric_loss
```

#### Key Parameters
- **input_dim**: Number of input features (varies based on dataset)
- **hidden_dim**: 64 (default)
- **dropout**: 0.1
- **learning_rate**: 0.001
- **λ1**: 0.5 (weight for SCA loss)
- **λ2**: 1.0 (weight for hypsometric loss)
- **optimizer**: Adam
- **scheduler**: ReduceLROnPlateau (factor=0.5, patience=50)
- **patience**: 300 (early stopping)

#### Key Features
- Snow Covered Area (SCA) computation
- Elevation-based SWE computation using binning
- Custom physics loss with multiple components
- Early stopping with patience
- Gradient clipping
- Non-negative SWE prediction enforcement

### Model Training and Evaluation

The `model_creation_gnn.py` script provides a comprehensive training and evaluation pipeline:

```bash
python code/model_creation_gnn.py
```

Key features:
- Custom loss function combining MSE and Smooth L1
- Learning rate scheduling
- Model evaluation with multiple metrics (MSE, R², RMSE, MAE)
- Early stopping

## Code Structure

### Core Files

- **Data Processing**:
  - `graph_data_prep.py`: Basic graph data preparation
  - `graph_data_creation.py`: Enhanced graph creation with temporal features
  - `graph_creation.py`: Advanced graph construction with adaptive thresholds
  - `state_data_creation.py`: State-specific data creation

- **Models**:
  - `graph_model.py`: Basic GraphSAGE implementation
  - `graph_model_sage.py`: Enhanced GraphSAGE with residual connections
  - `temporal_graph.py`: Physics-informed neural network implementation
  - `model_creation_gnn.py`: Comprehensive model training pipeline

- **Utilities**:
  - `fetch_data.sh`: Data download script
  - `install_modules.sh`: Dependency installation script

## Best Practices

### Data Processing

1. **Filtering**: Always filter data to relevant geographic regions
2. **Normalization**: Standardize features before graph creation
3. **Edge Creation**: Use adaptive thresholds for edge creation based on data distribution
4. **Graph Pruning**: Remove low-degree nodes to improve model performance

### Model Training

1. **Learning Rate**: Start with a small learning rate (1e-4 to 1e-3)
2. **Regularization**: Use dropout (0.1-0.4) to prevent overfitting
3. **Batch Normalization**: Apply after each convolutional layer
4. **Gradient Clipping**: Use to prevent exploding gradients
5. **Early Stopping**: Implement with patience to prevent overfitting

### Evaluation

1. **Multiple Metrics**: Use R², RMSE, and MAE for comprehensive evaluation
2. **Validation Split**: Use 20% of data for validation
3. **Scaling**: Remember to inverse transform predictions before computing metrics

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or model complexity
2. **Poor Convergence**: Adjust learning rate or try different optimizers
3. **Overfitting**: Increase dropout or add L2 regularization
4. **Missing Features**: Check CSV column names match expected columns

### Debugging Tips

1. Print graph statistics (nodes, edges, features) to verify graph creation
2. Monitor training loss and validation metrics
3. Check for NaN values in features or labels
4. Verify edge creation with visualization if possible

## Advanced Topics

### Physics-Informed Neural Networks

The `temporal_graph.py` implements a Physics-Informed Neural Network (PINN) approach that incorporates domain knowledge through custom loss functions:

1. **SCA Computation**: Calculates Snow Covered Area from SWE predictions
2. **Elevation Binning**: Groups SWE predictions by elevation bins
3. **Custom Loss**: Combines data-driven loss with physics-based constraints

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **Learning Rate**: 1e-4 to 1e-3
2. **Hidden Dimensions**: 64 to 512
3. **Dropout Rate**: 0.1 to 0.4
4. **Number of Layers**: 3 to 5
5. **Physics Loss Weights**: λ1 and λ2 in PINN implementation

## Future Improvements

1. Implement cross-validation for more robust evaluation
2. Add attention mechanisms to improve model performance
3. Incorporate more physics-based constraints
4. Implement distributed training for larger datasets
5. Add visualization tools for model interpretation