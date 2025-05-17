import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Dummy data loading function (replace with real data loading)
def load_data():
    # Assume data has been preprocessed into numpy arrays
    # X: [num_samples, num_features]
    # y: [num_samples, 3] -> [ATA, ATD, Berth_No]
    data = pd.read_csv("your_combined_dataset.csv")  # replace with actual path

    feature_cols = ['Type', 'Size', 'Draft', 'ETA', 'ETD', 'Congestion', 
                    'Weather', 'Reliability', 'Effectiveness', 'Work_Environment', 'Priority_Of_Shipment']
    label_cols = ['ATA', 'ATD', 'Berth_No']

    X = data[feature_cols].values.astype(np.float32)
    y = data[label_cols].values.astype(np.float32)

    return torch.tensor(X), torch.tensor(y)

# Define neural network model
class BerthAllocationModel(nn.Module):
    def __init__(self, input_dim):
        super(BerthAllocationModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Predicting ATA, ATD, Berth_No
        )

    def forward(self, x):
        return self.model(x)

# Fast overlap penalty loss

def fast_overlap_penalty(preds, penalty_weight=1.0):
    ATA = preds[:, 0]
    ATD = preds[:, 1]
    Berth = torch.round(preds[:, 2])

    ATA_i = ATA.unsqueeze(0)
    ATD_i = ATD.unsqueeze(0)
    Berth_i = Berth.unsqueeze(0)

    ATA_j = ATA.unsqueeze(1)
    ATD_j = ATD.unsqueeze(1)
    Berth_j = Berth.unsqueeze(1)

    same_berth = (Berth_i == Berth_j)
    overlap = (ATA_i < ATD_j) & (ATA_j < ATD_i)
    mask = ~torch.eye(preds.shape[0], dtype=torch.bool, device=preds.device)

    penalty_matrix = same_berth & overlap & mask
    overlap_time = torch.relu(torch.min(ATD_i, ATD_j) - torch.max(ATA_i, ATA_j))
    total_penalty = overlap_time[penalty_matrix].sum()

    return penalty_weight * total_penalty / preds.shape[0]

# Combined loss

def custom_loss_fast(y_pred, y_true, penalty_weight=0.1):
    mse = nn.MSELoss()(y_pred, y_true)
    overlap = fast_overlap_penalty(y_pred, penalty_weight)
    return mse + overlap

# Main training script
def train():
    X, y = load_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = BerthAllocationModel(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = custom_loss_fast(preds, batch_y, penalty_weight=0.1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "berth_allocation_model.pth")
    print("Model saved as berth_allocation_model.pth")

if __name__ == "__main__":
    train()