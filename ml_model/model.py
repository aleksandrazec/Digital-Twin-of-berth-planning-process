import torch
import torch.nn as nn

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


    def map_tensor_to_berth_range(tensor, min_val=1, max_val=50):
        clipped = torch.clamp(tensor, min=min_val, max=max_val)
        mapped = torch.round(clipped)
        return mapped.long()  # returns integer tensor

    # def map_tensor_to_1_50(self, tensor):
    #     # tanh gives values in [-1, 1]
    #     scaled = (torch.tanh(tensor) + 1) / 2  # Now in [0, 1]
    #     mapped = torch.floor(scaled * 49) + 1  # Now in [1, 50]
    #     return mapped.clamp(1, 50).long()
    
    def forward(self, x):
        out = self.model(x)
        etd = out[:, 0]
        eta = out[:, 1]

        # Generate a random integer berth between 1 and 50 for each sample
        batch_size = x.size(0)
        berth_random = torch.randint(1, 51, (batch_size,), device=x.device, dtype=torch.float)

        final_output = torch.stack([eta, etd, berth_random], dim=1)
        return final_output

def fast_overlap_penalty(preds, penalty_weight=1.0):
    ETA = preds[:, 0]
    ETD = preds[:, 1]
    Berth = torch.round(preds[:, 2])

    Berth_i = Berth.unsqueeze(0)
    ETA_i = ETA.unsqueeze(0)
    ETD_i = ETD.unsqueeze(0)

    Berth_j = Berth.unsqueeze(1)
    ETA_j = ETA.unsqueeze(1)
    ETD_j = ETD.unsqueeze(1)

    same_berth = (Berth_i == Berth_j)
    overlap = (ETA_i < ETD_j) & (ETA_j < ETD_i)
    mask = ~torch.eye(preds.shape[0], dtype=torch.bool, device=preds.device)

    penalty_matrix = same_berth & overlap & mask
    overlap_time = torch.relu(torch.min(ETD_i, ETD_j) - torch.max(ETA_i, ETA_j))
    total_penalty = overlap_time[penalty_matrix].sum()

    return penalty_weight * total_penalty / preds.shape[0]


def custom_loss_fast(y_pred, y_true, penalty_weight=0.1):
    mse = nn.MSELoss()(y_pred, y_true)
    overlap = fast_overlap_penalty(y_pred, penalty_weight)
    return mse + overlap