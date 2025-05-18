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
        self.berth_head = nn.Linear(32, 50) # 50 berth classes
        self.eta_etd_head = nn.Sequential(
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.shared(x)
        berth_logits = self.berth_head(x)
        eta_etd = self.eta_etd_head(x)
        return (berth_logits, eta_etd)

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

def custom_loss_fast(y_pred, y_true, penalty_weight=0.1, invalid_berth_weight=10.0):
    
    berth_logits, eta_etd_pred = y_pred
    
    berth_logits_true = y_true[:, 0]
    eta_etd_true = torch.cat([y_true[:, 1], y_true[:, 2]]) 
    
    berth_loss = nn.CrossEntropyLoss()(berth_logits.long(), berth_logits_true.long())

    mse = nn.MSELoss()(eta_etd_pred, eta_etd_true)
    preds_combined = torch.cat([
            torch.argmax(berth_logits, dim=1).unsqueeze(1).float(),  # predicted BERTH
            eta_etd_pred
    ], dim=1)

    overlap = fast_overlap_penalty(preds_combined, penalty_weight)


    return berth_loss + mse + overlap