"""
Full evaluation script with Test-Time Augmentation (TTA).

Supports:
  - Teacher (Swin-Base UNet++) and Student (EfficientNet-B2) models
  - Standard single-pass evaluation
  - TTA (8 augmentations averaged) for +1-2% mIoU boost
  - Both old and new model architectures

Usage:
    python evaluation/evaluate.py --model_type teacher --checkpoint checkpoints/teacher_best.pth
    python evaluation/evaluate.py --model_type teacher --checkpoint checkpoints/teacher_best.pth --tta
    python evaluation/evaluate.py --model_type student --checkpoint checkpoints/student_best.pth
"""

import sys
import os
import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.mbrsc_dataset import MBRSCDataset, get_val_transforms, CLASS_NAMES
from evaluation.metrics import (
    compute_miou,
    compute_accuracy,
    compute_per_class_iou,
    compute_f1,
    StreamingConfusionMatrix,
)


def load_model(model_type, checkpoint, device):
    """Load model based on type, trying new architecture first, falling back to old."""
    if model_type == "teacher":
        try:
            from models.teacher.swin_unet_plus import SwinUNetPlusPlus
            model = SwinUNetPlusPlus(num_classes=6, pretrained=False, deep_supervision=True)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            # Handle checkpoint dict vs raw state_dict
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            print(f"Loaded NEW teacher (Swin-Base UNet++) from {checkpoint}")
            return model.to(device)
        except Exception as e:
            print(f"Could not load as new teacher: {e}")
            print("Trying old teacher architecture...")
            from models.teacher.swin_unet import SwinUNetTeacher
            model = SwinUNetTeacher(num_classes=6, pretrained=False)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded OLD teacher (Swin-Tiny) from {checkpoint}")
            return model.to(device)
    else:
        try:
            from models.student.efficientnet_unet import EfficientNetUNet
            model = EfficientNetUNet(num_classes=6, pretrained=False)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded NEW student (EfficientNet-B2) from {checkpoint}")
            return model.to(device)
        except Exception as e:
            print(f"Could not load as new student: {e}")
            print("Trying old student architecture...")
            from models.student.mobilenet_unet import StudentUNet
            model = StudentUNet(num_classes=6, pretrained=False)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded OLD student (MobileNetV3) from {checkpoint}")
            return model.to(device)


def predict_with_tta(model, imgs, device):
    """
    Test-Time Augmentation: average predictions over 8 flips/rotations.
    Typically gives +1–2% mIoU over single-pass.
    """
    model.eval()
    imgs = imgs.to(device)
    preds = []

    augments = [
        lambda x: x,                                         # original
        lambda x: torch.flip(x, dims=[3]),                  # H-flip
        lambda x: torch.flip(x, dims=[2]),                  # V-flip
        lambda x: torch.flip(x, dims=[2, 3]),               # HV-flip
        lambda x: torch.rot90(x, 1, dims=[2, 3]),           # rot 90
        lambda x: torch.rot90(x, 2, dims=[2, 3]),           # rot 180
        lambda x: torch.rot90(x, 3, dims=[2, 3]),           # rot 270
        lambda x: torch.flip(torch.rot90(x, 1, dims=[2, 3]), dims=[3]),  # rot+flip
    ]
    inverse = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.flip(x, dims=[2, 3]),
        lambda x: torch.rot90(x, 3, dims=[2, 3]),
        lambda x: torch.rot90(x, 2, dims=[2, 3]),
        lambda x: torch.rot90(x, 1, dims=[2, 3]),
        lambda x: torch.rot90(torch.flip(x, dims=[3]), 3, dims=[2, 3]),
    ]

    with torch.no_grad():
        for aug, inv in zip(augments, inverse):
            aug_imgs = aug(imgs)
            out = model(aug_imgs)
            if isinstance(out, (tuple, list)):
                out = out[0]   # use main output only
            out = inv(out)
            preds.append(F.softmax(out, dim=1))

    return torch.stack(preds).mean(0).argmax(dim=1)


def evaluate_with_tta(model, loader, num_classes, device):
    """Evaluate model with test-time augmentation."""
    cm = StreamingConfusionMatrix(num_classes)
    model.eval()
    for imgs, masks in loader:
        preds = predict_with_tta(model, imgs, device)
        cm.update(preds.cpu(), masks)
    return cm.miou(), cm.accuracy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def measure_inference_speed(model, device, input_size=(1, 3, 224, 224), n_runs=100):
    """Measure average inference time in ms."""
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs * 1000
    return elapsed


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model (auto-detects old vs new architecture)
    model = load_model(args.model_type, args.checkpoint, device)
    model.eval()

    # Data
    val_ds = MBRSCDataset(
        img_dir=os.path.join(args.data_dir, "images"),
        mask_dir=os.path.join(args.data_dir, "masks"),
        split_file=args.split_file,
        transform=get_val_transforms(),
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    # Standard metrics
    miou = compute_miou(model, val_loader, 6, device)
    acc = compute_accuracy(model, val_loader, device)
    f1 = compute_f1(model, val_loader, 6, device)
    per_class = compute_per_class_iou(model, val_loader, 6, CLASS_NAMES, device)

    print(f"\n{'='*50}")
    print(f"Model       : {args.model_type}")
    print(f"mIoU        : {miou:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"F1 (macro)  : {f1:.4f}")
    print(f"Parameters  : {count_parameters(model) / 1e6:.2f}M")

    speed = measure_inference_speed(model, device)
    print(f"Inference   : {speed:.1f} ms/image")

    # Model size on disk
    ckpt_size = os.path.getsize(args.checkpoint) / 1e6
    print(f"Checkpoint  : {ckpt_size:.1f} MB")

    # TTA evaluation
    if args.tta:
        print(f"\n{'='*50}")
        print("Running TTA evaluation (8 augmentations)...")
        tta_miou, tta_acc = evaluate_with_tta(model, val_loader, 6, device)
        print(f"TTA mIoU    : {tta_miou:.4f}  (standard: {miou:.4f}, delta: +{tta_miou-miou:.4f})")
        print(f"TTA Accuracy: {tta_acc:.4f}  (standard: {acc:.4f}, delta: +{tta_acc-acc:.4f})")

    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--model_type", choices=["teacher", "student"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--split_file", type=str, default="data/splits/val.txt")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    args = parser.parse_args()
    evaluate(args)
