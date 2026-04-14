"""
Teacher training script for the Swin-Base UNet++ model.
Supports CUDA with mixed precision (AMP) for RTX 3050 compatibility.

Features:
  - AdamW optimizer with differential learning rates (encoder vs decoder)
  - Linear warmup + cosine annealing schedule
  - AMP (Automatic Mixed Precision) for VRAM savings on RTX 3050
  - Gradient clipping for training stability
  - Deep supervision (auxiliary losses from each decoder level)
  - Combined Focal + Dice + CE loss with class weights

Usage:
    python training/train_teacher.py
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
from datasets.mbrsc_dataset import MBRSCDataset, get_train_transforms, get_val_transforms
from training.losses import TeacherSegLoss
from evaluation.metrics import compute_miou, compute_accuracy


def build_dataloaders(batch_size=4, num_workers=2):
    """Build train and val dataloaders."""
    train_ds = MBRSCDataset(
        "data/processed/images", "data/processed/masks",
        "data/splits/train.txt", get_train_transforms()
    )
    val_ds = MBRSCDataset(
        "data/processed/images", "data/processed/masks",
        "data/splits/val.txt", get_val_transforms()
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    print(f"Train: {len(train_ds)} patches | Val: {len(val_ds)} patches")
    return train_loader, val_loader


def get_lr_scheduler(optimizer, total_epochs=150, warmup_epochs=10):
    """Linear warmup for first 10 epochs, then cosine annealing"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_teacher(batch_size=4, total_epochs=150):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
              if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
              else f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Model ────────────────────────────────────────────────────────────────
    model = SwinUNetPlusPlus(num_classes=6, pretrained=True, deep_supervision=True)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Teacher params: {total_params/1e6:.1f}M")

    # ── Class weights ────────────────────────────────────────────────────────
    try:
        class_weights = np.load("configs/class_weights.npy").tolist()
        print(f"Loaded class weights: {[f'{w:.2f}' for w in class_weights]}")
    except FileNotFoundError:
        print("Warning: class_weights.npy not found, using uniform weights.")
        print("Run: python training/compute_class_weights.py first!")
        class_weights = None

    criterion = TeacherSegLoss(num_classes=6, class_weights=class_weights, device=str(device))

    # ── Optimizer: AdamW (better weight decay than Adam for transformers) ────
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-5},  # lower LR for pretrained
            {"params": [p for n, p in model.named_parameters()
                        if "encoder" not in n], "lr": 1e-4},      # higher LR for decoder
        ],
        weight_decay=1e-2
    )
    scheduler = get_lr_scheduler(optimizer, total_epochs=total_epochs, warmup_epochs=10)

    # ── Data ─────────────────────────────────────────────────────────────────
    num_workers = 2 if device.type == "cuda" else 0
    train_loader, val_loader = build_dataloaders(batch_size=batch_size, num_workers=num_workers)

    # ── AMP scaler for CUDA ──────────────────────────────────────────────────
    scaler = GradScaler("cuda") if use_amp else None

    writer = SummaryWriter("runs/teacher_v2")
    best_miou, best_acc = 0.0, 0.0

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(total_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=False)

        for imgs, masks in pbar:
            imgs  = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if use_amp:
                with autocast("cuda"):
                    outputs = model(imgs)       # returns (main, aux3, aux2, aux1) during training
                    loss = criterion(outputs, masks)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        current_lr = optimizer.param_groups[1]["lr"]

        # ── Validate every 5 epochs ────────────────────────────────────────
        if (epoch + 1) % 5 == 0 or epoch == 0:
            miou = compute_miou(model, val_loader, num_classes=6, device=device)
            acc  = compute_accuracy(model, val_loader, device=device)

            writer.add_scalar("Loss/train",   total_loss / len(train_loader), epoch)
            writer.add_scalar("mIoU/val",     miou, epoch)
            writer.add_scalar("Accuracy/val", acc,  epoch)
            writer.add_scalar("LR",           current_lr, epoch)

            if device.type == "cuda":
                vram = torch.cuda.max_memory_allocated() / 1e9
                writer.add_scalar("VRAM_GB", vram, epoch)
                vram_str = f" | VRAM: {vram:.1f}GB"
            else:
                vram_str = ""

            print(f"Epoch [{epoch+1:3d}/{total_epochs}] "
                  f"Loss: {total_loss/len(train_loader):.4f}  "
                  f"mIoU: {miou:.4f}  Acc: {acc:.4f}  LR: {current_lr:.2e}{vram_str}")

            # Save best by mIoU
            if miou > best_miou:
                best_miou = miou
                torch.save(model.state_dict(), "checkpoints/teacher_best.pth")
                print(f"  ✓ Best teacher saved (mIoU={best_miou:.4f}, Acc={acc:.4f})")

            # Also save best by accuracy
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "checkpoints/teacher_best_acc.pth")

        # Save periodic checkpoint every 25 epochs
        if (epoch + 1) % 25 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'best_acc': best_acc,
            }, f"checkpoints/teacher_epoch_{epoch+1}.pth")

    writer.close()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"Teacher Training Complete!")
    print(f"Best mIoU:     {best_miou:.4f}  (target ≥ 0.82)")
    print(f"Best Accuracy: {best_acc:.4f}  (target ≥ 0.95)")

    if best_miou < 0.82:
        print("⚠ mIoU target not met. Try: longer training (200 epochs) or 384×384 patches.")
    if best_acc < 0.95:
        print("⚠ Accuracy target not met. Re-check class weights computation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Swin-Base UNet++ teacher")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default 4, reduce to 2 if OOM on RTX 3050)")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Total training epochs (default 150)")
    args = parser.parse_args()
    train_teacher(batch_size=args.batch_size, total_epochs=args.epochs)
