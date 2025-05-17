import pandas as pd
import numpy as np
import torch

def load_data(filepath="your_combined_dataset.csv"):
    data = pd.read_csv(filepath)  # Replace with your actual file path

    feature_cols = ['Type', 'Size', 'Draft', 'ETA', 'ETD', 'Congestion', 
                    'Weather', 'Reliability', 'Effectiveness', 'Work_Environment', 'Priority_Of_Shipment']
    label_cols = ['ATA', 'ATD', 'Berth_No']

    X = data[feature_cols].values.astype(np.float32)
    y = data[label_cols].values.astype(np.float32)

    return torch.tensor(X), torch.tensor(y)