"""
Loss functions for teacher training and knowledge distillation.

Components:
  - FocalLoss: Down-weights easy pixels, focuses on hard minority classes
  - DiceLoss: Directly optimises IoU-like overlap
  - TeacherSegLoss: Combined CE + Focal + Dice with deep supervision support
  - KnowledgeDistillationLoss: CE + Focal + Dice (hard) + KL (soft) + Feature MSE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Building blocks ────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal loss — down-weights easy pixels, focuses on hard minority classes"""
    def __init__(self, gamma=2.0, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    """Soft Dice loss — directly optimises IoU-like overlap"""
    def __init__(self, num_classes=6, smooth=1e-6, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        mask = (targets != self.ignore_index)
        dice = 0.0
        for c in range(self.num_classes):
            pred = probs[:, c][mask]
            true = (targets[mask] == c).float()
            dice += (2 * (pred * true).sum() + self.smooth) / \
                    (pred.sum() + true.sum() + self.smooth)
        return 1.0 - dice / self.num_classes


class TeacherSegLoss(nn.Module):
    """
    Combined loss for teacher training:
        Total = 0.4 * CE + 0.4 * Focal + 0.2 * Dice
    + Auxiliary deep supervision losses (weighted 0.4 each)
    """
    def __init__(self, num_classes=6, class_weights=None, device='cuda'):
        super().__init__()
        w = torch.FloatTensor(class_weights).to(device) if class_weights else None
        self.ce     = nn.CrossEntropyLoss(weight=w, ignore_index=255)
        self.focal  = FocalLoss(gamma=2.0, weight=w, ignore_index=255)
        self.dice   = DiceLoss(num_classes=num_classes)

    def _base(self, logits, targets):
        return 0.4 * self.ce(logits, targets) \
             + 0.4 * self.focal(logits, targets) \
             + 0.2 * self.dice(logits, targets)

    def forward(self, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            main, *aux = outputs
            loss = self._base(main, targets)
            for a in aux:
                loss = loss + 0.4 * self._base(a, targets)  # deep supervision
            return loss
        return self._base(outputs, targets)


# ── Knowledge Distillation Loss ────────────────────────────────────────────────


class KnowledgeDistillationLoss(nn.Module):
    """Combined loss for knowledge distillation.

    Total = β·(CE+Focal+Dice)(student, labels) + α·KL(student_soft, teacher_soft) + γ·MSE(feats)

    Parameters
    ----------
    temperature : float
        Softens logits for KL computation (higher = softer).
    alpha : float
        Weight for KL divergence (soft label) loss.
    beta : float
        Weight for cross-entropy (hard label) loss.
    gamma : float
        Weight for feature mimicking (MSE) loss.
    num_classes : int
        Number of segmentation classes.
    class_weights : list, optional
        Per-class weights for cross-entropy to handle class imbalance.
    device : str
        Device to place class weights on.
    """
    def __init__(self, temperature=4.0, alpha=0.5, beta=0.3, gamma=0.2,
                 num_classes=6, class_weights=None, device='cuda'):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        w = torch.FloatTensor(class_weights).to(device) if class_weights else None
        self.ce    = nn.CrossEntropyLoss(weight=w, ignore_index=255)
        self.focal = FocalLoss(gamma=2.0, weight=w)
        self.dice  = DiceLoss(num_classes=num_classes)

    def soft_label_loss(self, s_logits, t_logits):
        """KL divergence between temperature-softened distributions."""
        s = F.log_softmax(s_logits / self.T, dim=1)
        t = F.softmax(t_logits / self.T, dim=1)
        return F.kl_div(s, t, reduction='batchmean') * (self.T ** 2)

    def feature_loss(self, s_feats: dict, t_feats: dict):
        """MSE between aligned intermediate feature maps."""
        loss = 0
        for k in s_feats:
            if k in t_feats:
                sf = s_feats[k]
                tf = F.interpolate(t_feats[k].detach(), size=sf.shape[2:],
                                   mode='bilinear', align_corners=False)
                loss += F.mse_loss(sf, tf)
        return loss / max(len(s_feats), 1)

    def forward(self, s_logits, s_feats, t_logits, t_feats, labels):
        """
        Returns
        -------
        total_loss : Tensor
        log_dict : dict  (for TensorBoard logging)
        """
        hard = 0.4*self.ce(s_logits, labels) + 0.4*self.focal(s_logits, labels) \
             + 0.2*self.dice(s_logits, labels)
        kl   = self.soft_label_loss(s_logits, t_logits.detach())
        feat = self.feature_loss(s_feats, t_feats)
        total = self.beta * hard + self.alpha * kl + self.gamma * feat
        return total, {"ce_dice": hard.item(), "kl": kl.item(), "feat": feat.item()}
