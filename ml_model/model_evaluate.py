import torch
import numpy as np

import sys
import os

from model import BerthAllocationModel
from data import load_data

def evaluate(base_plan_file="dataset/berth_assignments_strict.csv",
              human_adjusted_file="dataset/augmented_berth_assignments_strict.csv",
              data_input_for_model_file="dataset/synthetic_estimated_final_full.csv",
             split_part=0.8
            ):
    # Load test data saved during training
    X, y = load_data(base_plan_file, human_adjusted_file, data_input_for_model_file)
    SPLIT = int(split_part * len(X))
    X_test, y_test = X[SPLIT:], y[SPLIT:]


    model = BerthAllocationModel(input_dim=X_test.shape[1])
    model.load_state_dict(torch.load("berth_allocation_model.pth"))
    model.eval()

    with torch.no_grad():
        test_preds = model(X_test)
        mse_loss = torch.nn.MSELoss()(test_preds, y_test)
        print(f"\nTest MSE Loss: {mse_loss.item():.4f}")

        test_preds_np = test_preds.numpy()
        test_preds_np[:, 0] = np.round(test_preds_np[:, 0])

        y_test_np = y_test.numpy()

        output_size = 50
        print(f"\nFirst {output_size} Test Predictions vs True Labels:")
        for i in range(min(output_size, len(test_preds_np))):
            print(f"Predicted: Berth_no={int(test_preds_np[i,2])}, ETA={test_preds_np[i,1]:.2f}, ETD={test_preds_np[i,0]} | "
                  f"True: Berth_no={int(y_test_np[i,2])}, ETA={y_test_np[i,1]:.2f}, ETD={y_test_np[i,0]}")

if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.exists(f"{sys.argv[1]}.csv"):
        print(f"Testing data on {sys.argv[1]}.csv")
        evaluate(sys.argv[1])
    else:
        print("For testing your own data provide only 1 command line argument:\n\t-the path of the .csv file containing the data")
        evaluate()