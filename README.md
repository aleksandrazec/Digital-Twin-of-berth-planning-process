import torch
import torch.nn as nn
import torch.nn.functional as F

class BerthAllocationModel(nn.Module):
    def __init__(self, input_dim):
        super(BerthAllocationModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.output_berth = nn.Linear(32, 50)  # Output logits for berth classes 1–50
        self.output_time = nn.Linear(32, 2)    # Outputs: ATA, ATD (continuous)

    def forward(self, x):
        x = self.shared(x)

        # Berth prediction as classification
        berth_logits = self.output_berth(x)  # shape: (batch_size, 50)
        berth_probs = F.softmax(berth_logits, dim=1)
        berth_pred = torch.argmax(berth_probs, dim=1).float() + 1  # range [1, 50]

        # ATA/ATD prediction as regression
        ata_atd = self.output_time(x)  # shape: (batch_size, 2)

        # Combine outputs: [berth_number, ATA, ATD]
        output = torch.cat([berth_pred.unsqueeze(1), ata_atd], dim=1)
        return output, berth_logits  # logits needed for classification loss


# ⛔ Overlap Penalty Function (same as before)
def fast_overlap_penalty(preds, penalty_weight=1.0):
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


# ✅ Custom Loss Function
def custom_loss(y_pred_tuple, y_true, penalty_weight=0.1):
    y_pred, berth_logits = y_pred_tuple
    # True values
    true_berth = y_true[:, 0].long() - 1  # convert to [0–49] class index
    true_ata_atd = y_true[:, 1:]

    # Cross-entropy for berth classification
    berth_loss = nn.CrossEntropyLoss()(berth_logits, true_berth)

    # MSE for ATA and ATD
    ata_atd_loss = nn.MSELoss()(y_pred[:, 1:], true_ata_atd)

    # Overlap penalty
    overlap = fast_overlap_penalty(y_pred, penalty_weight)

    # Total loss
    return berth_loss + ata_atd_loss + overlap
