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

    def forward(self, x):
        return self.model(x)

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

def custom_loss_fast(y_pred, y_true, penalty_weight=0.1):
    mse = nn.MSELoss()(y_pred, y_true)
    overlap = fast_overlap_penalty(y_pred, penalty_weight)
    return mse + overlap