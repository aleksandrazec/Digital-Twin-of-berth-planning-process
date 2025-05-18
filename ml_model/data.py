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

def load_data(base_plan_file="dataset/berth_assignments_strict.csv",
              human_adjusted_file="dataset/augmented_berth_assignments_strict.csv",
              data_input_for_model_file="dataset/synthetic_estimated_final_full.csv"
            ):
    # data = pd.read_csv(filepath)  # Replace with your actual file path

    # feature_cols = ['Type', 'Size', 'Draft', 'ETA', 'ETD', 'Congestion', 
    #                 'Weather', 'Reliability', 'Effectiveness', 'Work_Environment', 'Priority_Of_Shipment']
    # label_cols = ['ATA', 'ATD', 'Berth_No']

    

    data, feature_cols, label_cols = get_data(  base_plan_file, human_adjusted_file, data_input_for_model_file) 
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 10)

    # print(data[feature_cols])
    # print(data[label_cols])

    # feature_cols = [col for col in feature_cols if col != "BASE_BERTH"]
    # label_cols = [col for col in label_cols if col != "H_BERTH"]

    X = data[feature_cols].values.astype(np.float32)
    y = data[label_cols].values.astype(np.float32)

    

    return torch.tensor(X), torch.tensor(y)