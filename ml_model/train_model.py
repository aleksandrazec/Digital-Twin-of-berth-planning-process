import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.model_selection import train_test_split

from data import load_data
from model import BerthAllocationModel, custom_loss_fast

def train(  base_plan_file="dataset/berth_assignments_strict.csv",
              human_adjusted_file="dataset/augmented_berth_assignments_strict.csv",
              data_input_for_model_file="dataset/synthetic_estimated_final_full.csv",
            split_part=0.8
            ):
    X, y = load_data(base_plan_file, human_adjusted_file, data_input_for_model_file)

    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_part, random_state=42)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = BerthAllocationModel(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = custom_loss_fast(preds, batch_y, penalty_weight=0.1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    # Save model and test data for evaluation
    torch.save(model.state_dict(), "berth_allocation_model.pth")
    torch.save((X_test, y_test), "test_data.pt")
    print("\nModel saved as berth_allocation_model.pth")
    print("Test data saved as test_data.pt")

if __name__ == "__main__":
    train()