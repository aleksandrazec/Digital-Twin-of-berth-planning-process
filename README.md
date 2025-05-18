import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

def load_data():
    data, feature_cols, label_cols = get_data()

    # Create LabelEncoders for BASE_BERTH and H_BERTH
    le_base_berth = LabelEncoder()
    le_h_berth = LabelEncoder()

    # Encode berth strings to integer classes
    data["BASE_BERTH_enc"] = le_base_berth.fit_transform(data["BASE_BERTH"])
    data["H_BERTH_enc"] = le_h_berth.fit_transform(data["H_BERTH"])

    # Replace berth columns in feature_cols and label_cols with encoded versions
    feature_cols_mod = feature_cols.copy()
    label_cols_mod = label_cols.copy()

    # Replace original berth columns with encoded ones
    feature_cols_mod = [col for col in feature_cols_mod if col != "BASE_BERTH"]
    feature_cols_mod.append("BASE_BERTH_enc")

    label_cols_mod = [col for col in label_cols_mod if col != "H_BERTH"]
    label_cols_mod.append("H_BERTH_enc")

    # Prepare feature tensor (float for continuous + int for berth)
    X_float = data[[col for col in feature_cols_mod if col != "BASE_BERTH_enc"]].values.astype(np.float32)
    X_berth = data["BASE_BERTH_enc"].values.astype(np.int64)

    # Combine continuous features and berth class into a single tensor (float + int)
    # We cannot combine int and float easily in one tensor, so pass them separately to model
    X_float_tensor = torch.tensor(X_float)
    X_berth_tensor = torch.tensor(X_berth, dtype=torch.long)

    # Prepare label tensor
    y_float = data[[col for col in label_cols_mod if col != "H_BERTH_enc"]].values.astype(np.float32)
    y_berth = data["H_BERTH_enc"].values.astype(np.int64)

    y_float_tensor = torch.tensor(y_float)
    y_berth_tensor = torch.tensor(y_berth, dtype=torch.long)

    return (X_float_tensor, X_berth_tensor), (y_float_tensor, y_berth_tensor), le_base_berth, le_h_berth



import torch
import torch.nn as nn
import torch.nn.functional as F

class BerthAllocationModel(nn.Module):
    def __init__(self, input_dim_float, num_berth_classes):
        super(BerthAllocationModel, self).__init__()
        
        # Embedding for berth input feature
        self.berth_embedding = nn.Embedding(num_berth_classes, embedding_dim=8)
        
        # Fully connected layers for float features
        self.fc_float = nn.Sequential(
            nn.Linear(input_dim_float, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combine embedded berth + processed float features
        self.fc_combined = nn.Sequential(
            nn.Linear(32 + 8, 64),
            nn.ReLU(),
        )
        
        # Output heads:
        # 1) ETA and ETD (continuous regression)
        self.out_times = nn.Linear(64, 2)  # predict ETA and ETD
        
        # 2) Berth classification (number of classes)
        self.out_berth = nn.Linear(64, num_berth_classes)
    
    def forward(self, x_float, x_berth):
        # x_float: (batch_size, float_features)
        # x_berth: (batch_size,) int64 tensor for berth class
        
        berth_embedded = self.berth_embedding(x_berth)  # (batch_size, embedding_dim)
        
        float_feat = self.fc_float(x_float)  # (batch_size, 32)
        
        combined = torch.cat([float_feat, berth_embedded], dim=1)  # (batch_size, 40)
        
        combined_feat = self.fc_combined(combined)  # (batch_size, 64)
        
        times_pred = self.out_times(combined_feat)  # (batch_size, 2) float
        berth_logits = self.out_berth(combined_feat)  # (batch_size, num_berth_classes)
        
        return times_pred, berth_logits


mse_loss = nn.MSELoss()
ce_loss = nn.CrossEntropyLoss()

def combined_loss(times_pred, times_true, berth_logits, berth_true):
    loss_times = mse_loss(times_pred, times_true)
    loss_berth = ce_loss(berth_logits, berth_true)
    return loss_times + loss_berth
