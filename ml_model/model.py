import torch
import torch.nn as nn

class BerthAllocationModel(nn.Module):
    def __init__(self, input_dim):
        super(BerthAllocationModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.berth_head = nn.Linear(32, 50)  # 50 berth classes (1 to 50)
        self.eta_etd_head = nn.Sequential(
            nn.Linear(32, 2)  # Predict H_ETA and H_ETD
        )

    def forward(self, x):
        x = self.shared(x)
        berth_logits = self.berth_head(x)  # raw logits
        eta_etd = self.eta_etd_head(x)
        predicted_berth = torch.argmax(berth_logits, dim=1, keepdim=True).float() + 1  # shape: (batch_size, 1)

        return  torch.cat([predicted_berth, eta_etd], dim=1)


def fast_overlap_penalty(preds, penalty_weight=1.0):
    # preds must be a [batch_size, 3] tensor: [BERTH, ETA, ETD]
    Berth = torch.round(preds[:, 0])
    ETA = preds[:, 1]
    ETD = preds[:, 2]

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
    # y_pred is [batch_size, 3]: [BERTH (predicted as class index + 1), ETA, ETD]
    # y_true is also [batch_size, 3]: [BERTH (true, in 1–50), ETA, ETD]


    mse = nn.MSELoss()(y_pred, y_true)

    # Overlap penalty
    overlap = fast_overlap_penalty(y_pred, penalty_weight)

    return mse + overlap
