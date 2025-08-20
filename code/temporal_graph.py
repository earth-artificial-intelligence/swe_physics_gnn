import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau


# SET YOUR GRAPH FILE PATH HERE
GRAPH_DATA_PATH = "/media/volume1/gnn_graph_pinn_data_new_3.pt"
EPOCHS = 2000
LAMBDA1 = 0.5
LAMBDA2 = 1.0
TEST_RATIO = 0.2

# 1️⃣ PINN Model
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout=0.1):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x).squeeze()

# 2️⃣ Compute SCA (Snow Covered Area)
def compute_sca(swe_pred, threshold=0.01):
    return (swe_pred > threshold).float().mean()

# 3️⃣ Compute SWE by Elevation Bins
def compute_swe_by_elevation(swe_pred, elevation, bins):
    digitized = torch.bucketize(elevation, bins)
    means = [swe_pred[digitized == i].mean() if (digitized == i).any()
             else torch.tensor(0.0, device=swe_pred.device) for i in range(1, len(bins))]
    return torch.stack(means)

# 4️⃣ Custom Loss
def pinn_physics_loss(pred, y_true, sca_pred, sca_ref, swe_model_by_elev, swe_climo_by_elev, λ1=1.0, λ2=1.0):
    model_loss = F.mse_loss(pred, y_true)
    sca_loss = F.l1_loss(sca_pred, sca_ref)
    hypsometric_loss = F.l1_loss(swe_model_by_elev, swe_climo_by_elev)
    return model_loss + λ1 * sca_loss + λ2 * hypsometric_loss

# 5️⃣ Split Graph Data


def split_graph_data(graph_data, test_ratio=0.2, seed=None):
    """
    Splits graph nodes into train and test indices with a fixed test ratio.

    Parameters:
        graph_data: torch_geometric.data.Data
        test_ratio: float
        seed: int (optional)

    Returns:
        train_idx, test_idx: torch.Tensor, torch.Tensor
    """
    if seed is not None:
        torch.manual_seed(seed)

    num_nodes = graph_data.num_nodes
    num_test = int(round(test_ratio * num_nodes))  # round to ensure fixed size
    indices = torch.randperm(num_nodes)

    test_idx = indices[:num_test]
    train_idx = indices[num_test:]

    return train_idx, test_idx



# 6️⃣ Train Function

def train_pinn_with_improvements(graph_data, epochs=10000, λ1=1.0, λ2=1.0, test_ratio=0.2, patience=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = graph_data.x.shape[1]
    model = PINN(input_dim=input_dim, hidden_dim=64, dropout=0.1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

    graph_data = graph_data.to(device)
    elevation = graph_data.x[:, 0]
    elevation_bins = torch.tensor([0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000], device=device)
    sca_ref = torch.tensor(0.6, device=device)
    swe_climo = torch.tensor(
    [0.1, 0.15, 0.25, 0.35, 0.45, 0.55, 0.60, 0.55, 0.50, 0.40, 0.30, 0.20],
    device=device)

    
    

    train_idx, val_idx = split_graph_data(graph_data, test_ratio)

    best_val_loss = float('inf')
    best_r2 = -1
    best_model_state = None
    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        pred_train = model(graph_data.x[train_idx])
        y_train = graph_data.y[train_idx]
        elev_train = elevation[train_idx]

        sca_pred = compute_sca(pred_train)
        swe_model_hypso = compute_swe_by_elevation(pred_train, elev_train, elevation_bins)

        loss = pinn_physics_loss(pred_train, y_train, sca_pred, sca_ref, swe_model_hypso, swe_climo, λ1, λ2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(graph_data.x[val_idx])
            y_val = graph_data.y[val_idx]
            val_loss = F.mse_loss(pred_val, y_val)
            r2 = r2_score(y_val.cpu().numpy(), pred_val.cpu().numpy())

        scheduler.step(val_loss)

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | R²: {r2:.4f}")

        scheduler.step(val_loss)

         # ✅ Track best R² and best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            best_r2 = r2
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch} — Best R²: {best_r2:.4f}")
            break

    model.load_state_dict(best_model_state)

    # Final Eval
    model.eval()
    with torch.no_grad():
        pred = model(graph_data.x[val_idx]).cpu().numpy()
        pred = np.clip(pred, 0, None)  # ⛔ Prevent negative SWE predictions
        true = graph_data.y[val_idx].cpu().numpy()

        final_mse = F.mse_loss(torch.tensor(pred), torch.tensor(true)).item()
        final_mae = F.l1_loss(torch.tensor(pred), torch.tensor(true)).item()
        final_r2 = r2_score(true, pred)


    print("\n✅ Final Evaluation:")
    print(f"📉 Final MSE      : {final_mse:.4f}")
    print(f"📊 Final MAE      : {final_mae:.4f}")
    print(f"📈 Final R² Score : {final_r2:.4f}")
    print("🏁 Model training & evaluation complete.\n")

    # 🔍 Show a few prediction vs actual pairs
    print("🔍 Sample predictions vs actual:")
    for i in range(5):
        error = abs(pred[i] - true[i])
        print(f"   Predicted: {pred[i]:.3f} | Actual: {true[i]:.3f} | Error: {error:.3f}")


    return model

# evaulate the model 
def evaluate_pinn(model, graph_data, test_ratio=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    graph_data = graph_data.to(device)

    _, val_idx = split_graph_data(graph_data, test_ratio)
    model.eval()

    with torch.no_grad():
        pred = model(graph_data.x[val_idx]).cpu().numpy()
        pred = np.clip(pred, 0, None)
        true = graph_data.y[val_idx].cpu().numpy()

        mse = F.mse_loss(torch.tensor(pred), torch.tensor(true)).item()
        mae = F.l1_loss(torch.tensor(pred), torch.tensor(true)).item()
        r2 = r2_score(true, pred)

        print("\n🔍 Test Set Evaluation:")
        print(f"📉 MSE : {mse:.4f}")
        print(f"📊 MAE : {mae:.4f}")
        print(f"📈 R² Score : {r2:.4f}")

        print("\n🔍 Sample predictions vs actual:")
        for i in range(10):
            error = abs(pred[i] - true[i])
            print(f"   Predicted: {pred[i]:.3f} | Actual: {true[i]:.3f} | Error: {error:.3f}")




# 7️⃣ Just Run This
# 7️⃣ Just Run This
if __name__ == "__main__":
    print(f"📥 Loading graph from: {GRAPH_DATA_PATH}")
    graph_data = torch.load(GRAPH_DATA_PATH, weights_only=False)

    # 📊 Inspect target SWE values
    print("\n📊 Inspecting graph_data.y:")
    print("   Min :", graph_data.y.min().item())
    print("   Max :", graph_data.y.max().item())
    print("   Mean:", graph_data.y.mean().item())
    print("   Std :", graph_data.y.std().item())
    print("   Sample values:", graph_data.y[:10])

    print(f"🚀 Training PINN for {EPOCHS} epochs...")
    trained_model = train_pinn_with_improvements(
        graph_data=graph_data,
        epochs=EPOCHS,
        λ1=LAMBDA1,
        λ2=LAMBDA2,
        test_ratio=TEST_RATIO
    )



