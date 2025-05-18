import torch
import numpy as np

import sys
import os

from model import BerthAllocationModel
from data import load_data

def evaluate(dataFile=None):
    # Load test data saved during training
    if dataFile is None:
        X, y = load_data()
        SPLIT = int(0.8 * len(X))
        X_test, y_test = X[SPLIT:], y[SPLIT:]
    else:
        X_test, y_test = load_data(dataFile)


    model = BerthAllocationModel(input_dim=X_test.shape[1])
    model.load_state_dict(torch.load("berth_allocation_model.pth"))
    model.eval()

    with torch.no_grad():
        test_preds = model(X_test)
        mse_loss = torch.nn.MSELoss()(test_preds, y_test)
        print(f"\nTest MSE Loss: {mse_loss.item():.4f}")

        test_preds_np = test_preds.numpy()
        test_preds_np[:, 2] = np.round(test_preds_np[:, 2])

        y_test_np = y_test.numpy()

        output_size = 100
        print(f"\nFirst {output_size} Test Predictions vs True Labels:")
        for i in range(min(output_size, len(test_preds_np))):
            print(f"Predicted: ATA={test_preds_np[i,0]:.2f}, ATD={test_preds_np[i,1]:.2f}, Berth_No={int(test_preds_np[i,2])} | "
                  f"True: ATA={y_test_np[i,0]:.2f}, ATD={y_test_np[i,1]:.2f}, Berth_No={int(y_test_np[i,2])}")

if __name__ == "__main__":
    if len(sys.argv) == 2 and os.path.exists(f"{sys.argv[1]}.csv"):
        print(f"Testing data on {sys.argv[1]}.csv")
        evaluate(sys.argv[1])
    else:
        print("For testing your own data provide only 1 command line argument:\n\t-the path of the .csv file containing the data")
        evaluate()