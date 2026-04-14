"""
Evaluation metrics for semantic segmentation.

Provides mIoU, pixel accuracy, per-class IoU, and F1 score computation.
"""

import torch
import numpy as np


@torch.no_grad()
def compute_miou(model, loader, num_classes, device):
    """Compute mean Intersection-over-Union on a dataloader."""
    model.eval()
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(dim=1)

        for c in range(num_classes):
            pred_c = preds == c
            true_c = masks == c
            intersection[c] += (pred_c & true_c).sum()
            union[c] += (pred_c | true_c).sum()

    iou_per_class = intersection.float() / (union.float() + 1e-8)
    return iou_per_class.mean().item()


@torch.no_grad()
def compute_accuracy(model, loader, device):
    """Compute overall pixel accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(dim=1)
        correct += (preds == masks).sum().item()
        total += masks.numel()

    return correct / total


@torch.no_grad()
def compute_per_class_iou(model, loader, num_classes, class_names, device):
    """Compute and print per-class IoU."""
    model.eval()
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(dim=1)

        for c in range(num_classes):
            pred_c = preds == c
            true_c = masks == c
            intersection[c] += (pred_c & true_c).sum()
            union[c] += (pred_c | true_c).sum()

    iou = intersection.float() / (union.float() + 1e-8)

    print("\nPer-Class IoU:")
    for i, name in enumerate(class_names):
        print(f"  {name:<12}: {iou[i].item():.4f}")
    print(f"  {'mIoU':<12}: {iou.mean().item():.4f}")
    return iou.cpu()


@torch.no_grad()
def compute_f1(model, loader, num_classes, device):
    """Compute macro-averaged F1 score."""
    model.eval()
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = logits.argmax(dim=1)

        for c in range(num_classes):
            pred_c = preds == c
            true_c = masks == c
            tp[c] += (pred_c & true_c).sum()
            fp[c] += (pred_c & ~true_c).sum()
            fn[c] += (~pred_c & true_c).sum()

    precision = tp.float() / (tp.float() + fp.float() + 1e-8)
    recall = tp.float() / (tp.float() + fn.float() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


class StreamingConfusionMatrix:
    """Accumulates predictions across batches for accurate epoch-level mIoU"""
    def __init__(self, num_classes):
        self.C = num_classes
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.long)

    def update(self, preds, labels):
        mask = (labels >= 0) & (labels < self.C)
        idx = self.C * labels[mask] + preds[mask]
        self.mat += torch.bincount(idx, minlength=self.C**2).reshape(self.C, self.C)

    def miou(self):
        diag = self.mat.diag()
        union = self.mat.sum(1) + self.mat.sum(0) - diag
        iou = diag.float() / (union.float() + 1e-8)
        return iou[union > 0].mean().item()

    def accuracy(self):
        return (self.mat.diag().sum() / self.mat.sum()).item()
