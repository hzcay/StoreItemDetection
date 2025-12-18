import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)

    This implementation assumes:
    - features: (B, D) already L2-normalized (recommended)
    - labels:   (B,) int64 class ids for positives (e.g., SKU id)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.dim() != 2:
            raise ValueError(f"features must be 2D (B, D), got {tuple(features.shape)}")
        if labels.dim() != 1:
            labels = labels.view(-1)

        device = features.device
        B = features.size(0)
        if B <= 1:
            # No contrastive pairs
            return features.new_tensor(0.0)

        labels = labels.to(device)

        # Similarity logits (B, B)
        logits = (features @ features.T) / self.temperature

        # For numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # Mask: positives if same label, exclude self
        matches = labels.unsqueeze(0).eq(labels.unsqueeze(1))  # (B, B)
        self_mask = torch.eye(B, device=device, dtype=torch.bool)
        positives = matches & (~self_mask)

        # exp logits excluding self
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Mean log-likelihood over positives for each anchor
        pos_counts = positives.sum(dim=1)  # (B,)
        # Avoid division by zero: anchors without positives contribute 0
        mean_log_prob_pos = torch.zeros(B, device=device, dtype=features.dtype)
        valid = pos_counts > 0
        if valid.any():
            mean_log_prob_pos[valid] = (log_prob[valid] * positives[valid].float()).sum(dim=1) / (
                pos_counts[valid].float() + 1e-12
            )

        loss = -mean_log_prob_pos[valid].mean() if valid.any() else features.new_tensor(0.0)
        return loss


