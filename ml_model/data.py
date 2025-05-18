import pandas as pd
import numpy as np
import torch

import sys
sys.path.append('../data processing/')
from get_data import get_data 



# TODO - change the data
# the data needs to have beside vessel attributes, weather, congestion, reliabality
# a couple of different ATAs, ATDs
# 

def load_data(filepath="your_combined_dataset.csv"):
    # data = pd.read_csv(filepath)  # Replace with your actual file path

    # feature_cols = ['Type', 'Size', 'Draft', 'ETA', 'ETD', 'Congestion', 
    #                 'Weather', 'Reliability', 'Effectiveness', 'Work_Environment', 'Priority_Of_Shipment']
    # label_cols = ['ATA', 'ATD', 'Berth_No']

    

    data, feature_cols, label_cols = get_data() 
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    print(data[feature_cols])
    print(data[label_cols])

    

    X = data[feature_cols].values.astype(np.float32)
    y = data[label_cols].values.astype(np.float32)

    return torch.tensor(X), torch.tensor(y)