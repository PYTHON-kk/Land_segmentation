"""
Knowledge Distillation training script.
Trains EfficientNet-B2 student using frozen Swin-Base UNet++ teacher.

Supports CUDA with AMP for RTX 3050 compatibility.

Usage:
    python training/train_student_kd.py
"""

import sys
import os
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.teacher.swin_unet_plus import SwinUNetPlusPlus
from models.student.efficientnet_unet import EfficientNetUNet
from datasets.mbrsc_dataset import MBRSCDataset, get_train_transforms, get_val_transforms
from training.losses import KnowledgeDistillationLoss
from evaluation.metrics import compute_miou, compute_accuracy

# Teacher feature hooks
teacher_features = {}


def make_hook(name):
    def hook(module, input, output):
        teacher_features[name] = output.detach()
    return hook


def train_student_kd(batch_size=8, total_epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load frozen teacher ───────────────────────────────────────────────────
    teacher = SwinUNetPlusPlus(num_classes=6, pretrained=False, deep_supervision=True)
    state = torch.load("checkpoints/teacher_best.pth", map_location=device, weights_only=True)
    # Handle checkpoint dict vs raw state_dict
    if isinstance(state, dict) and 'model_state_dict' in state:
        teacher.load_state_dict(state['model_state_dict'])
    else:
        teacher.load_state_dict(state)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Register hooks on teacher decoder nodes to extract intermediate features
    teacher.node_0_1.register_forward_hook(make_hook("d3"))
    teacher.node_0_2.register_forward_hook(make_hook("d2"))
    teacher.node_0_3.register_forward_hook(make_hook("d1"))
    print("Teacher loaded and frozen. Hooks registered.")

    # ── Student ───────────────────────────────────────────────────────────────
    student = EfficientNetUNet(num_classes=6, pretrained=True).to(device)
    s_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Student params: {s_params/1e6:.1f}M")

    # ── Class weights ──────────────────────────────────────────────────────────
    try:
        class_weights = np.load("configs/class_weights.npy").tolist()
        print(f"Loaded class weights: {[f'{w:.2f}' for w in class_weights]}")
    except FileNotFoundError:
        print("Warning: class_weights.npy not found, using uniform weights.")
        class_weights = None

    kd_loss_fn = KnowledgeDistillationLoss(
        temperature=4.0, alpha=0.5, beta=0.3, gamma=0.2,
        num_classes=6, class_weights=class_weights, device=str(device)
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # ── Data ─────────────────────────────────────────────────────────────────
    num_workers = 2 if device.type == "cuda" else 0
    train_ds = MBRSCDataset("data/processed/images", "data/processed/masks",
                             "data/splits/train.txt", get_train_transforms())
    val_ds   = MBRSCDataset("data/processed/images", "data/processed/masks",
                             "data/splits/val.txt", get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    print(f"Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")

    # ── AMP scaler ───────────────────────────────────────────────────────────
    scaler = GradScaler("cuda") if use_amp else None

    writer = SummaryWriter("runs/student_kd_v2")
    best_miou = 0.0

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(total_epochs):
        student.train()
        total_loss, logs_accum = 0.0, {"ce_dice": 0, "kl": 0, "feat": 0}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            # Teacher forward (fills teacher_features via hooks)
            with torch.no_grad():
                t_out = teacher(imgs)
                t_logits = t_out if not isinstance(t_out, tuple) else t_out[0]
                t_feats = dict(teacher_features)  # snapshot

            if use_amp:
                with autocast("cuda"):
                    s_logits, s_feats = student(imgs, return_features=True)
                    loss, logs = kd_loss_fn(s_logits, s_feats, t_logits, t_feats, masks)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                s_logits, s_feats = student(imgs, return_features=True)
                loss, logs = kd_loss_fn(s_logits, s_feats, t_logits, t_feats, masks)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            for k in logs_accum:
                logs_accum[k] += logs[k]
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            miou = compute_miou(student, val_loader, num_classes=6, device=device)
            acc  = compute_accuracy(student, val_loader, device=device)
            n    = len(train_loader)
            writer.add_scalar("Loss/total",    total_loss / n, epoch)
            writer.add_scalar("Loss/kl",       logs_accum["kl"] / n, epoch)
            writer.add_scalar("Loss/feat",     logs_accum["feat"] / n, epoch)
            writer.add_scalar("mIoU/val",      miou, epoch)
            writer.add_scalar("Accuracy/val",  acc,  epoch)

            if device.type == "cuda":
                vram = torch.cuda.max_memory_allocated() / 1e9
                vram_str = f" | VRAM: {vram:.1f}GB"
            else:
                vram_str = ""

            print(f"Epoch [{epoch+1:3d}/{total_epochs}] mIoU: {miou:.4f}  Acc: {acc:.4f}  "
                  f"Loss: {total_loss/n:.4f}{vram_str}")
            if miou > best_miou:
                best_miou = miou
                torch.save(student.state_dict(), "checkpoints/student_best.pth")
                print(f"  ✓ Best student saved (mIoU={best_miou:.4f}, Acc={acc:.4f})")

    writer.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\nKD Complete. Best Student mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student via knowledge distillation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (default 8, reduce if OOM)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs (default 100)")
    args = parser.parse_args()
    train_student_kd(batch_size=args.batch_size, total_epochs=args.epochs)
