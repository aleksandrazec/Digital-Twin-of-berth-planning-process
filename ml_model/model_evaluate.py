import torch
import numpy as np

from model import BerthAllocationModel

def evaluate():
    # Load test data saved during training
    X_test, y_test = torch.load("test_data.pt")

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

        print("\nFirst 10 Test Predictions vs True Labels:")
        for i in range(min(10, len(test_preds_np))):
            print(f"Predicted: ATA={test_preds_np[i,0]:.2f}, ATD={test_preds_np[i,1]:.2f}, Berth_No={int(test_preds_np[i,2])} | "
                  f"True: ATA={y_test_np[i,0]:.2f}, ATD={y_test_np[i,1]:.2f}, Berth_No={int(y_test_np[i,2])}")

if __name__ == "__main__":
    evaluate()