"""Quick dry-run: verify the full pipeline is ready for training."""
import torch
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.teacher.swin_unet_plus import SwinUNetPlusPlus
from models.student.efficientnet_unet import EfficientNetUNet
from datasets.mbrsc_dataset import MBRSCDataset, get_train_transforms, get_val_transforms
from training.losses import TeacherSegLoss, KnowledgeDistillationLoss
from evaluation.metrics import compute_miou, compute_accuracy, StreamingConfusionMatrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Test data loading
train_ds = MBRSCDataset("data/processed/images", "data/processed/masks",
                        "data/splits/train.txt", get_train_transforms())
val_ds = MBRSCDataset("data/processed/images", "data/processed/masks",
                      "data/splits/val.txt", get_val_transforms())
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# Test class weights
cw = np.load("configs/class_weights.npy").tolist()
print(f"Class weights loaded: {[f'{w:.2f}' for w in cw]}")

# Test full teacher loss with weights
loss_fn = TeacherSegLoss(num_classes=6, class_weights=cw, device=str(device))
print("TeacherSegLoss initialized with class weights")

# Test teacher model with pretrained weights
print("Loading teacher with pretrained=True (downloads Swin-Base weights)...")
teacher = SwinUNetPlusPlus(num_classes=6, pretrained=True, deep_supervision=True).to(device)
print(f"Teacher ready: {sum(p.numel() for p in teacher.parameters())/1e6:.1f}M params")

# Quick forward pass with real data
img, mask = train_ds[0]
img = img.unsqueeze(0).to(device)
teacher.train()
with torch.amp.autocast("cuda"):
    out = teacher(img)
    mask_t = mask.unsqueeze(0).to(device)
    loss = loss_fn(out, mask_t)
print(f"Forward pass loss: {loss.item():.4f}")

# Check VRAM
if device.type == "cuda":
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("\n" + "="*50)
print("All verified! Ready for training.")
print("Run: python training/train_teacher.py")
print("="*50)
