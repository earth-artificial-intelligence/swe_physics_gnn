import numpy as np
import torch
import warnings
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import SAGEConv, BatchNorm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import add_self_loops

warnings.filterwarnings("ignore")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Loss Function
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)
        self.alpha = alpha

    def forward(self, predictions, targets):
        return self.alpha * self.mse(predictions, targets) + (1 - self.alpha) * self.smooth_l1(predictions, targets)

# GraphSAGE Model
class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, dropout=0.2):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr="max")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="max")
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr="max")
        self.conv4 = SAGEConv(hidden_dim, hidden_dim, aggr="max")
        self.conv5 = SAGEConv(hidden_dim, hidden_dim, aggr="max")

        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.bn4 = BatchNorm(hidden_dim)
        self.bn5 = BatchNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.conv4(x, edge_index)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn5(self.conv5(x, edge_index)))
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load Graph
def load_graph(path):
    from torch_geometric.data import Data
    from torch.serialization import add_safe_globals
    add_safe_globals([Data])  

    with open(path, 'rb') as f:
        g_data = torch.load(f, weights_only=False)  
    print(f"✅ Loaded graph from {path}")
    print(f"Nodes: {g_data.num_nodes}, Edges: {g_data.num_edges}, Features: {g_data.x.shape}")
    return g_data



# Train Step
def train(model, optimizer, loss_fn, data, mask):
    model.train()
    optimizer.zero_grad()
    preds = model(data)[mask].squeeze()
    loss = loss_fn(preds, data.y[mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return loss.item()

# Test Step
def test(model, loss_fn, data, mask, y_scaler):
    model.eval()
    with torch.no_grad():
        preds = model(data)[mask].squeeze().cpu().numpy()
        actuals = data.y[mask].cpu().numpy()

        preds_original = y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        actuals_original = y_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

        pd.DataFrame({
            'predicted_swe': preds_original,
            'actual_swe': actuals_original
        }).to_csv("predicted_vs_actual.csv", index=False)

        mse = mean_squared_error(actuals_original, preds_original)
        r2 = r2_score(actuals_original, preds_original)
        mae = mean_absolute_error(actuals_original, preds_original)
        rmse = mean_squared_error(actuals_original, preds_original, squared=False)

    return mse, r2, rmse, mae

# Main
def main():
    g_path = "/media/volume1/gnn_graph_with_time_all_states_2.pt"
    data = load_graph(g_path)

    # Normalize features
    x_scaler = MinMaxScaler()
    data.x = torch.tensor(x_scaler.fit_transform(data.x.cpu()), dtype=torch.float).to(device)

    # Normalize labels
    y_scaler = MinMaxScaler()
    data.y = torch.tensor(y_scaler.fit_transform(data.y.view(-1, 1)), dtype=torch.float).view(-1).to(device)

    # Add self-loops
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes)

    # Train/Test Split
    indices = torch.randperm(data.num_nodes)
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[indices[:int(0.8 * data.num_nodes)]] = True
    test_mask = ~train_mask

    print(f"🔁 Training on {train_mask.sum().item()}, Testing on {test_mask.sum().item()}")

    model = GraphSAGE(input_dim=data.num_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=25)
    criterion = CustomLoss(alpha=0.7)

    best_r2 = -float("inf")
    wait = 0
    patience = 200

    print("🚀 Starting training...")
    for epoch in range(1, 2001):
        loss = train(model, optimizer, criterion, data, train_mask)

        if epoch % 100 == 0 or epoch == 1:
            mse, r2, rmse, mae = test(model, criterion, data, test_mask, y_scaler)
            print(f"Epoch {epoch} ➤ Train Loss: {loss:.4f} | R²: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                print(f"🛑 Early stopping at epoch {epoch}")
                break

    print(f"\n✅ Best R²: {best_r2:.4f}")

if __name__ == "__main__":
    main()
