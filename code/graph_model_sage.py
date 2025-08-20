import numpy as np
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.utils import add_self_loops
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Check Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and verify graph
def load_and_verify_graph(g_path):
    print("Loading graph dataset")
    try:
        g_data = torch.load(g_path)
        print(f"Graph successfully loaded from {g_path}")
        print(f"Number of Nodes: {g_data.num_nodes}")
        print(f"Number of Edges: {g_data.num_edges}")
        print(f"Node Feature Shape: {g_data.x.shape}")
        print(f"Edge Index Shape: {g_data.edge_index.shape}")

        if hasattr(g_data, "y") and g_data.y is not None:
            print(f"Target Variable Shape: {g_data.y.shape}")
        else:
            print(f"Warning: Target variable y is missing or None!")

        return g_data
    except Exception as e:
        print(f"Error loading graph: {e}")
        exit()

# Define GraphSAGE model with residual connections
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

        self.dropout1 = nn.Dropout(0.2)  # Slightly higher dropout in early layers
        self.dropout2 = nn.Dropout(0.1)  # Lower dropout in final layers

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Project input features to hidden_dim for residual connections
        x = self.input_proj(x)

        residual = x  # Keep projected features for residual connection

        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout2(x)
        
        x = x + residual  # Residual connection

        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training function
def train(model, optimizer, loss_fn, graph, train_mask):
    model.train()
    optimizer.zero_grad()
    predictions = model(graph)[train_mask].squeeze()
    loss = loss_fn(predictions, graph.y[train_mask])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)  # Improved gradient clipping
    optimizer.step()
    return loss.item()

# Testing function
def test(model, loss_fn, graph, test_mask):
    model.eval()
    with torch.no_grad():
        predictions = model(graph)[test_mask].squeeze().cpu().numpy()
        actuals = graph.y[test_mask].cpu().numpy()

        mse = mean_squared_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = mean_squared_error(actuals, predictions, squared=False)

    return mse, r2, rmse, mae

# Main function
def main():
    g_path = "/media/volume1/gnn_2graph.pt"  
    g_data = load_and_verify_graph(g_path)

    # Normalize node features
    scaler_x = StandardScaler()
    g_data.x = torch.tensor(scaler_x.fit_transform(g_data.x.cpu()), dtype=torch.float).to(device)

    # Normalize target variable using StandardScaler (fixes scaling issues)
    y_scaler = StandardScaler()
    g_data.y = torch.tensor(y_scaler.fit_transform(g_data.y.view(-1, 1)), dtype=torch.float).view(-1).to(device)

    # Training split
    indices = np.arange(g_data.num_nodes)
    np.random.shuffle(indices)
    train_size = int(0.8 * len(indices))
    
    train_mask = torch.zeros(g_data.num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    test_mask = ~train_mask  

    print(f"Training on {train_size} nodes & Testing on {g_data.num_nodes - train_size} nodes")

    # Define model, optimizer, and loss function
    model = GraphSAGE(input_dim=g_data.num_features, hidden_dim=256, output_dim=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Slightly lower learning rate
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)  
    criterion = nn.SmoothL1Loss(beta=1.0)

    best_r2 = -float("inf")

    print("Starting training for 1000 epochs...")

    for epoch in range(1, 1001):  
        train_loss = train(model, optimizer, criterion, g_data, train_mask)

        if epoch % 100 == 0 or epoch == 5:  
            test_loss, r2, rmse, mae = test(model, criterion, g_data, test_mask)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f} | R²={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f}")

            # Update best R² score
            if r2 > best_r2:
                best_r2 = r2

        # Reduce learning rate dynamically
        scheduler.step()

    print(f"Best R² score: {best_r2:.4f}")

if __name__ == "__main__":
    main()
