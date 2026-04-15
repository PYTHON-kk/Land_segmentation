"""
Visualization module for land segmentation evaluation.

Generates publication-quality plots:
  1. Per-class IoU bar chart
  2. Confusion matrix heatmap (counts + normalized)
  3. Sample prediction gallery (image / GT / prediction overlay)
  4. Precision, Recall, F1 grouped bar chart
  5. Class pixel distribution (pie + bar)
    6. Model comparison plots (teacher vs student + optional paper baselines)

Usage:
    python evaluation/visualize.py --model_type teacher --checkpoint checkpoints/teacher_best.pth
    python evaluation/visualize.py --model_type student --checkpoint checkpoints/student_best.pth
    python evaluation/visualize.py --compare  (compares both models side by side)
"""

import sys
import os
import argparse
import json
import time
from itertools import cycle

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.mbrsc_dataset import MBRSCDataset, get_val_transforms, CLASS_NAMES, NUM_CLASSES
from evaluation.metrics import StreamingConfusionMatrix

# ─── Color Palette ────────────────────────────────────────────────────────────
# Vivid class colors for segmentation overlays
CLASS_COLORS = np.array([
    [238,  76,  44],   # building  – coral red
    [196, 167, 125],   # land      – sandy tan
    [ 70,  70,  70],   # road      – dark gray
    [ 34, 197,  94],   # vegetation – green
    [ 56, 182, 255],   # water     – sky blue
    [160, 160, 160],   # unlabeled – light gray
], dtype=np.uint8)

# Chart accent colors (one per class)
CHART_COLORS = ["#EE4C2C", "#C4A77D", "#464646", "#22C55E", "#38B6FF", "#A0A0A0"]

# Global plot style
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor":   "#FFFFFF",
    "axes.edgecolor":   "#D1D5DB",
    "axes.labelcolor":  "#111827",
    "text.color":       "#111827",
    "xtick.color":      "#111827",
    "ytick.color":      "#111827",
    "grid.color":       "#D1D5DB",
    "grid.alpha":       0.6,
    "font.family":      "sans-serif",
    "font.size":        11,
})


def load_paper_baselines(json_path):
    """Load optional paper baselines from JSON.

    Expected JSON format (list of objects):
    [
      {
        "name": "InceptionResNetV2-UNet (ICCAD 2025)",
        "miou": 0.78,
        "accuracy": 0.93,
        "f1_macro": 0.86,
        "building_iou": 0.80,
        "vegetation_iou": 0.84,
        "water_iou": 0.70
      }
    ]

    Only name + (miou, accuracy, f1_macro) are required for bar comparison.
    Class IoU fields are optional (used only in radar chart).
    """
    if not json_path:
        return []

    if not os.path.exists(json_path):
        print(f"⚠ Baseline file not found: {json_path}")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"⚠ Failed to read paper baseline JSON: {e}")
        return []

    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        print("⚠ Baseline JSON must be a list or object. Ignoring.")
        return []

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    baselines = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "Paper Model")
        miou = _num(item.get("miou", None))
        acc = _num(item.get("accuracy", None))
        f1m = _num(item.get("f1_macro", None))

        if miou is None or acc is None or f1m is None:
            print(f"⚠ Skipping baseline '{name}': requires miou, accuracy, f1_macro")
            continue

        baselines.append({
            "name": name,
            "miou": miou,
            "accuracy": acc,
            "f1_macro": f1m,
            "per_class_iou": np.array([
                _num(item.get("building_iou", np.nan)) if _num(item.get("building_iou", np.nan)) is not None else np.nan,
                np.nan,
                np.nan,
                _num(item.get("vegetation_iou", np.nan)) if _num(item.get("vegetation_iou", np.nan)) is not None else np.nan,
                _num(item.get("water_iou", np.nan)) if _num(item.get("water_iou", np.nan)) is not None else np.nan,
                np.nan,
            ], dtype=float),
        })

    if baselines:
        print(f"Loaded {len(baselines)} paper baseline(s) from: {json_path}")
    return baselines


# ─── Model Loading (reuse from evaluate.py) ──────────────────────────────────
def load_model(model_type, checkpoint, device):
    """Load model based on type, trying new architecture first."""
    if model_type == "teacher":
        try:
            from models.teacher.swin_unet_plus import SwinUNetPlusPlus
            model = SwinUNetPlusPlus(num_classes=6, pretrained=False, deep_supervision=True)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            print(f"Loaded teacher (Swin-Base UNet++) from {checkpoint}")
            return model.to(device)
        except Exception as e:
            print(f"Falling back to old teacher: {e}")
            from models.teacher.swin_unet import SwinUNetTeacher
            model = SwinUNetTeacher(num_classes=6, pretrained=False)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            return model.to(device)
    else:
        try:
            from models.student.efficientnet_unet import EfficientNetUNet
            model = EfficientNetUNet(num_classes=6, pretrained=False)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            print(f"Loaded student (EfficientNet-B2) from {checkpoint}")
            return model.to(device)
        except Exception as e:
            print(f"Falling back to old student: {e}")
            from models.student.mobilenet_unet import StudentUNet
            model = StudentUNet(num_classes=6, pretrained=False)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            return model.to(device)


# ─── Gather Full Evaluation Data ─────────────────────────────────────────────
@torch.no_grad()
def gather_eval_data(model, loader, num_classes, device):
    """
    Single pass over the val set to collect:
      - confusion matrix
      - per-class TP/FP/FN
      - a few sample images/masks/preds for visualization
    """
    model.eval()
    cm = StreamingConfusionMatrix(num_classes)
    tp = torch.zeros(num_classes, device=device)
    fp = torch.zeros(num_classes, device=device)
    fn = torch.zeros(num_classes, device=device)

    samples = []  # store a handful of (img, gt, pred) for the gallery
    max_samples = 8

    for imgs, masks in loader:
        imgs_d, masks_d = imgs.to(device), masks.to(device)
        logits = model(imgs_d)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        preds = logits.argmax(dim=1)

        cm.update(preds.cpu(), masks)

        for c in range(num_classes):
            pred_c = preds == c
            true_c = masks_d == c
            tp[c] += (pred_c & true_c).sum()
            fp[c] += (pred_c & ~true_c).sum()
            fn[c] += (~pred_c & true_c).sum()

        # Collect sample images (denormalize)
        if len(samples) < max_samples:
            for i in range(min(imgs.size(0), max_samples - len(samples))):
                img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
                # Denormalize from ImageNet stats
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                samples.append((img_np, masks[i].cpu().numpy(), preds[i].cpu().numpy()))

    # Compute per-class metrics
    precision = tp.float() / (tp.float() + fp.float() + 1e-8)
    recall = tp.float() / (tp.float() + fn.float() + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Per-class IoU from confusion matrix
    mat = cm.mat
    diag = mat.diag()
    union = mat.sum(1) + mat.sum(0) - diag
    iou = diag.float() / (union.float() + 1e-8)

    return {
        "confusion_matrix": mat.numpy(),
        "per_class_iou": iou.numpy(),
        "precision": precision.cpu().numpy(),
        "recall": recall.cpu().numpy(),
        "f1": f1.cpu().numpy(),
        "miou": iou[union > 0].mean().item(),
        "accuracy": (diag.sum() / mat.sum()).item(),
        "f1_macro": f1.mean().item(),
        "samples": samples,
        "class_pixel_counts": mat.sum(1).numpy(),  # actual pixels per class
    }


# ─── 1. Per-Class IoU Bar Chart ──────────────────────────────────────────────
def plot_per_class_iou(data, model_name, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))

    ious = data["per_class_iou"]
    x = np.arange(len(CLASS_NAMES))
    bars = ax.bar(x, ious * 100, color=CHART_COLORS, edgecolor="#D1D5DB",
                  width=0.6, zorder=3)

    # Add value labels on bars
    for bar, val in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val*100:.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=10, color="#111827")

    # mIoU horizontal line
    miou = data["miou"] * 100
    ax.axhline(y=miou, color="#FACC15", linewidth=2, linestyle="--", alpha=0.8, zorder=2)
    ax.text(len(CLASS_NAMES) - 0.5, miou + 2, f"mIoU: {miou:.1f}%",
            ha="right", fontweight="bold", color="#FACC15", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight="medium")
    ax.set_ylabel("IoU (%)", fontsize=12)
    ax.set_title(f"Per-Class IoU — {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─── 2. Confusion Matrix Heatmap ─────────────────────────────────────────────
def plot_confusion_matrix(data, model_name, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    mat = data["confusion_matrix"].astype(float)
    # Normalized version (row-wise = per true class)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = np.divide(mat, row_sums, where=row_sums != 0, out=np.zeros_like(mat))

    for ax, m, title, fmt in [
        (axes[0], mat,      "Confusion Matrix (Counts)", ".0f"),
        (axes[1], mat_norm, "Normalized Confusion Matrix", ".2f"),
    ]:
        im = ax.imshow(m, cmap="YlOrRd", aspect="equal")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        # Annotate cells
        thresh = m.max() / 2
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                val = m[i, j]
                text_val = f"{val:{fmt}}" if fmt == ".0f" else f"{val:{fmt}}"
                color = "#111827" if val <= thresh else "#FFFFFF"
                ax.text(j, i, text_val, ha="center", va="center",
                        fontsize=8, color=color, fontweight="medium")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors="#111827")

    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─── 3. Sample Predictions Gallery ───────────────────────────────────────────
def mask_to_color(mask):
    """Convert class-index mask to RGB color image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        color[mask == c] = CLASS_COLORS[c]
    return color


def plot_sample_predictions(data, model_name, save_path):
    samples = data["samples"]
    n = min(len(samples), 6)
    if n == 0:
        print("  ⚠ No samples to plot")
        return

    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input Image", "Ground Truth", "Model Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=10)

    for i in range(n):
        img, gt, pred = samples[i]
        gt_color = mask_to_color(gt)
        pred_color = mask_to_color(pred)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt_color)
        axes[i, 2].imshow(pred_color)

        for j in range(3):
            axes[i, j].axis("off")

    # Legend
    patches = []
    for i in range(NUM_CLASSES):
        hex_color = f"#{CLASS_COLORS[i][0]:02x}{CLASS_COLORS[i][1]:02x}{CLASS_COLORS[i][2]:02x}"
        patches.append(mpatches.Patch(facecolor=hex_color, edgecolor="#D1D5DB", label=CLASS_NAMES[i]))
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES,
               fontsize=10, framealpha=1.0, facecolor="#FFFFFF", edgecolor="#D1D5DB",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"Sample Predictions — {model_name}", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─── 4. Precision / Recall / F1 Grouped Bar Chart ────────────────────────────
def plot_precision_recall_f1(data, model_name, save_path):
    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    prec_bars = ax.bar(x - width, data["precision"] * 100, width,
                       label="Precision", color="#4F46E5", edgecolor="#D1D5DB", zorder=3)
    rec_bars  = ax.bar(x,         data["recall"] * 100,    width,
                       label="Recall",    color="#059669", edgecolor="#D1D5DB", zorder=3)
    f1_bars   = ax.bar(x + width, data["f1"] * 100,        width,
                       label="F1 Score",  color="#EA580C", edgecolor="#D1D5DB", zorder=3)

    # Value labels
    for bars in [prec_bars, rec_bars, f1_bars]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, color="#111827")

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, fontsize=11, fontweight="medium")
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title(f"Precision / Recall / F1 — {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10, loc="upper right", framealpha=1.0, facecolor="#FFFFFF", edgecolor="#D1D5DB")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─── 5. Class Distribution ───────────────────────────────────────────────────
def plot_class_distribution(data, model_name, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    counts = data["class_pixel_counts"]
    total = counts.sum()
    percentages = counts / total * 100

    # Donut chart
    wedges, texts, autotexts = ax1.pie(
        percentages, labels=CLASS_NAMES, autopct="%1.1f%%",
        colors=CHART_COLORS, startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="#FFFFFF", linewidth=2),
        textprops=dict(fontsize=9, color="#111827")
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight("bold")
    ax1.set_title("Class Pixel Distribution", fontsize=12, fontweight="bold", pad=15)

    # Horizontal bar chart
    y = np.arange(len(CLASS_NAMES))
    bars = ax2.barh(y, percentages, color=CHART_COLORS, edgecolor="#D1D5DB", height=0.55, zorder=3)
    for bar, pct in zip(bars, percentages):
        ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va="center", fontsize=10, color="#111827", fontweight="bold")
    ax2.set_yticks(y)
    ax2.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax2.set_xlabel("Percentage of Pixels (%)", fontsize=11)
    ax2.set_title("Class Imbalance Analysis", fontsize=12, fontweight="bold", pad=15)
    ax2.set_xlim(0, max(percentages) * 1.2)
    ax2.grid(axis="x", linestyle="--", alpha=0.3)
    ax2.invert_yaxis()

    fig.suptitle(f"Dataset Class Distribution — {model_name}", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─── 6. Model Comparison Radar Chart ─────────────────────────────────────────
def plot_model_comparison(teacher_data, student_data, save_path, paper_baselines=None):
    """Comparison plots with white background.

    Includes:
      1) Radar chart: teacher, student, and optional paper baselines.
      2) Grouped bars: mIoU / Accuracy / F1 across all compared models.
    """
    paper_baselines = paper_baselines or []

    models = [
        {
            "name": "Teacher (Swin-Base UNet++)",
            "metrics": teacher_data,
            "color": "#2563EB",
            "marker": "o",
        },
        {
            "name": "Student (EfficientNet-B2 UNet)",
            "metrics": student_data,
            "color": "#059669",
            "marker": "s",
        },
    ]

    palette = cycle(["#7C3AED", "#DC2626", "#EA580C", "#0EA5E9", "#9333EA", "#CA8A04"])
    for b in paper_baselines:
        models.append({
            "name": b["name"],
            "metrics": b,
            "color": next(palette),
            "marker": "^",
        })

    fig = plt.figure(figsize=(17, 7), facecolor="#FFFFFF")

    # ── Radar chart ──
    ax1 = fig.add_subplot(121, polar=True, facecolor="#FFFFFF")
    metrics = ["mIoU", "Accuracy", "F1 Score", "Building\nIoU", "Vegetation\nIoU", "Water\nIoU"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    for m in models:
        md = m["metrics"]
        vals = [
            float(md["miou"]),
            float(md["accuracy"]),
            float(md["f1_macro"]),
            float(md["per_class_iou"][0]) if not np.isnan(md["per_class_iou"][0]) else np.nan,
            float(md["per_class_iou"][3]) if not np.isnan(md["per_class_iou"][3]) else np.nan,
            float(md["per_class_iou"][4]) if not np.isnan(md["per_class_iou"][4]) else np.nan,
        ]

        # Radar needs complete vector; skip models missing class IoU entries.
        if np.isnan(np.array(vals)).any():
            print(f"  ⚠ Skipping radar line for {m['name']} (missing class IoU values)")
            continue

        vals_closed = vals + vals[:1]
        ax1.plot(angles_closed, vals_closed, f"{m['marker']}-", color=m["color"], linewidth=2.2,
                 markersize=6, label=m["name"], zorder=3)
        ax1.fill(angles_closed, vals_closed, color=m["color"], alpha=0.08)

    ax1.set_xticks(angles)
    ax1.set_xticklabels(metrics, fontsize=9, color="#111827")
    ax1.set_ylim(0, 1.05)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="#374151")
    ax1.spines["polar"].set_color("#D1D5DB")
    ax1.grid(color="#D1D5DB", alpha=0.7)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.45, 1.18), fontsize=9,
               framealpha=1.0, facecolor="#FFFFFF", edgecolor="#D1D5DB")
    ax1.set_title("Performance Radar", fontsize=13, fontweight="bold", pad=20, color="#111827")

    # ── Global metric grouped bars ──
    ax2 = fig.add_subplot(122, facecolor="#FFFFFF")
    metric_names = ["mIoU", "Accuracy", "F1 Macro"]
    x = np.arange(len(metric_names))
    n_models = len(models)
    width = 0.8 / max(n_models, 1)

    for i, m in enumerate(models):
        md = m["metrics"]
        vals = np.array([md["miou"], md["accuracy"], md["f1_macro"]], dtype=float) * 100.0
        offset = (i - (n_models - 1) / 2) * width
        bars = ax2.bar(x + offset, vals, width,
                       label=m["name"], color=m["color"], alpha=0.9,
                       edgecolor="#D1D5DB", zorder=3)
        for b in bars:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.1f}",
                     ha="center", va="bottom", fontsize=7, color="#111827")

    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_names, fontsize=10, fontweight="medium")
    ax2.set_ylabel("Score (%)", fontsize=11, color="#111827")
    ax2.set_title("Global Metrics Comparison", fontsize=13, fontweight="bold", pad=15, color="#111827")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=8.5, framealpha=1.0, facecolor="#FFFFFF", edgecolor="#D1D5DB", loc="lower right")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.tick_params(colors="#111827")

    fig.suptitle("Model Comparison (White Background)",
                 fontsize=16, fontweight="bold", color="#111827", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─── Summary Dashboard ───────────────────────────────────────────────────────
def plot_summary_dashboard(data, model_name, model_type, checkpoint, save_path):
    """Single-image dashboard summarizing all key metrics."""
    fig = plt.figure(figsize=(18, 10), facecolor="#FFFFFF")
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── Title bar ──
    fig.suptitle(f"Land Segmentation — {model_name} Evaluation Dashboard",
                 fontsize=18, fontweight="bold", color="#111827", y=0.98)

    # ── 1. Metrics summary cards (top-left) ──
    ax0 = fig.add_subplot(gs[0, 0], facecolor="#FFFFFF")
    ax0.axis("off")
    metrics_text = [
        ("mIoU",      f"{data['miou']*100:.1f}%"),
        ("Accuracy",  f"{data['accuracy']*100:.1f}%"),
        ("F1 (macro)",f"{data['f1_macro']*100:.1f}%"),
        ("Model",     model_type.title()),
    ]
    # Compute param count & checkpoint size
    ckpt_size = os.path.getsize(checkpoint) / 1e6 if os.path.exists(checkpoint) else 0
    metrics_text.append(("Checkpoint", f"{ckpt_size:.1f} MB"))

    for i, (label, value) in enumerate(metrics_text):
        y = 0.88 - i * 0.18
        ax0.text(0.05, y, label, fontsize=11, color="#374151", transform=ax0.transAxes,
                 fontweight="medium")
        ax0.text(0.95, y, value, fontsize=14, color="#FACC15", transform=ax0.transAxes,
                 fontweight="bold", ha="right")
        if i < len(metrics_text) - 1:
            ax0.plot([0.05, 0.95], [y - 0.06, y - 0.06], color="#D1D5DB",
                     linewidth=0.5, transform=ax0.transAxes)
    ax0.set_title("Key Metrics", fontsize=13, fontweight="bold", pad=12, color="#111827")

    # ── 2. Per-class IoU (top-center) ──
    ax1 = fig.add_subplot(gs[0, 1], facecolor="#FFFFFF")
    ious = data["per_class_iou"]
    x = np.arange(len(CLASS_NAMES))
    bars = ax1.bar(x, ious * 100, color=CHART_COLORS, edgecolor="#D1D5DB", width=0.6, zorder=3)
    for bar, val in zip(bars, ious):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{val*100:.1f}", ha="center", fontsize=8, color="#111827", fontweight="bold")
    ax1.axhline(y=data["miou"] * 100, color="#FACC15", linewidth=1.5, linestyle="--", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n[:4] for n in CLASS_NAMES], fontsize=8)
    ax1.set_ylabel("IoU %", fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.set_title("Per-Class IoU", fontsize=13, fontweight="bold", pad=12, color="#111827")
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # ── 3. Normalized confusion matrix (top-right) ──
    ax2 = fig.add_subplot(gs[0, 2], facecolor="#FFFFFF")
    mat = data["confusion_matrix"].astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = np.divide(mat, row_sums, where=row_sums != 0, out=np.zeros_like(mat))
    im = ax2.imshow(mat_norm, cmap="YlOrRd", aspect="equal", vmin=0, vmax=1)
    ax2.set_xticks(range(len(CLASS_NAMES)))
    ax2.set_yticks(range(len(CLASS_NAMES)))
    ax2.set_xticklabels([n[:4] for n in CLASS_NAMES], rotation=45, ha="right", fontsize=8)
    ax2.set_yticklabels([n[:4] for n in CLASS_NAMES], fontsize=8)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val = mat_norm[i, j]
            color = "#FFFFFF" if val > 0.5 else "#111827"
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax2.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12, color="#111827")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors="#111827")

    # ── 4. Sample prediction (bottom-left, spans 1 col) ──
    if data["samples"]:
        ax3 = fig.add_subplot(gs[1, 0], facecolor="#FFFFFF")
        ax3.axis("off")
        img, gt, pred = data["samples"][0]
        # Show all 3 side by side in one axes
        combined = np.concatenate([img, mask_to_color(gt), mask_to_color(pred)], axis=1)
        ax3.imshow(combined)
        ax3.set_title("Input  |  Ground Truth  |  Prediction", fontsize=11, fontweight="bold",
                      pad=10, color="#111827")

    # ── 5. Precision/Recall/F1 (bottom-center) ──
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#FFFFFF")
    width = 0.25
    ax4.bar(x - width, data["precision"] * 100, width, label="Prec", color="#4F46E5",
            edgecolor="#D1D5DB", zorder=3)
    ax4.bar(x, data["recall"] * 100, width, label="Rec", color="#059669",
            edgecolor="#D1D5DB", zorder=3)
    ax4.bar(x + width, data["f1"] * 100, width, label="F1", color="#EA580C",
            edgecolor="#D1D5DB", zorder=3)
    ax4.set_xticks(x)
    ax4.set_xticklabels([n[:4] for n in CLASS_NAMES], fontsize=8)
    ax4.set_ylabel("Score %", fontsize=9)
    ax4.set_ylim(0, 110)
    ax4.set_title("Precision / Recall / F1", fontsize=13, fontweight="bold", pad=12, color="#111827")
    ax4.legend(fontsize=8, loc="upper right", framealpha=1.0, facecolor="#FFFFFF", edgecolor="#D1D5DB")
    ax4.grid(axis="y", linestyle="--", alpha=0.3)

    # ── 6. Class distribution (bottom-right) ──
    ax5 = fig.add_subplot(gs[1, 2], facecolor="#FFFFFF")
    counts = data["class_pixel_counts"]
    pcts = counts / counts.sum() * 100
    bars = ax5.barh(np.arange(len(CLASS_NAMES)), pcts, color=CHART_COLORS,
                    edgecolor="#D1D5DB", height=0.5, zorder=3)
    for bar, pct in zip(bars, pcts):
        ax5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va="center", fontsize=8, color="#111827")
    ax5.set_yticks(np.arange(len(CLASS_NAMES)))
    ax5.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax5.set_xlabel("% Pixels", fontsize=9)
    ax5.set_title("Class Distribution", fontsize=13, fontweight="bold", pad=12, color="#111827")
    ax5.invert_yaxis()
    ax5.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument("--model_type", choices=["teacher", "student"],
                        help="Model type to evaluate")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--split_file", type=str, default="data/splits/val.txt")
    parser.add_argument("--output_dir", type=str, default="evaluation/results",
                        help="Directory to save plots")
    parser.add_argument("--compare", action="store_true",
                        help="Compare teacher vs student (uses default checkpoints)")
    parser.add_argument("--teacher_ckpt", type=str, default="checkpoints/teacher_best.pth")
    parser.add_argument("--student_ckpt", type=str, default="checkpoints/student_best.pth")
    parser.add_argument("--paper_baselines_json", type=str, default="",
                        help="Optional JSON file with paper baseline metrics")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data loader
    val_ds = MBRSCDataset(
        img_dir=os.path.join(args.data_dir, "images"),
        mask_dir=os.path.join(args.data_dir, "masks"),
        split_file=args.split_file,
        transform=get_val_transforms(),
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    print(f"Validation set: {len(val_ds)} patches\n")

    if args.compare:
        paper_baselines = load_paper_baselines(args.paper_baselines_json)

        # ── Compare both models ──
        print("━" * 50)
        print("Evaluating TEACHER...")
        print("━" * 50)
        teacher_model = load_model("teacher", args.teacher_ckpt, device)
        teacher_data = gather_eval_data(teacher_model, val_loader, NUM_CLASSES, device)
        del teacher_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"\n{'━' * 50}")
        print("Evaluating STUDENT...")
        print("━" * 50)
        student_model = load_model("student", args.student_ckpt, device)
        student_data = gather_eval_data(student_model, val_loader, NUM_CLASSES, device)
        del student_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Generate all plots for both
        for name, data, ckpt in [("Teacher", teacher_data, args.teacher_ckpt),
                                  ("Student", student_data, args.student_ckpt)]:
            tag = name.lower()
            print(f"\n📊 Generating plots for {name}...")
            plot_per_class_iou(data, name, os.path.join(args.output_dir, f"{tag}_per_class_iou.png"))
            plot_confusion_matrix(data, name, os.path.join(args.output_dir, f"{tag}_confusion_matrix.png"))
            plot_sample_predictions(data, name, os.path.join(args.output_dir, f"{tag}_sample_predictions.png"))
            plot_precision_recall_f1(data, name, os.path.join(args.output_dir, f"{tag}_precision_recall_f1.png"))
            plot_class_distribution(data, name, os.path.join(args.output_dir, f"{tag}_class_distribution.png"))
            plot_summary_dashboard(data, name, tag, ckpt,
                                   os.path.join(args.output_dir, f"{tag}_dashboard.png"))

        # Comparison chart
        print("\n📊 Generating comparison chart...")
        plot_model_comparison(teacher_data, student_data,
                              os.path.join(args.output_dir, "model_comparison.png"),
                              paper_baselines=paper_baselines)

        # Save metrics JSON
        summary = {
            "teacher": {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in teacher_data.items() if k != "samples"},
            "student": {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in student_data.items() if k != "samples"},
            "paper_baselines": [
                {
                    "name": b["name"],
                    "miou": b["miou"],
                    "accuracy": b["accuracy"],
                    "f1_macro": b["f1_macro"],
                }
                for b in paper_baselines
            ],
        }
        json_path = os.path.join(args.output_dir, "metrics_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved: {json_path}")

    else:
        # ── Single model evaluation ──
        if not args.model_type or not args.checkpoint:
            parser.error("--model_type and --checkpoint are required (or use --compare)")

        model_name = args.model_type.title()
        print(f"{'━' * 50}")
        print(f"Evaluating {model_name}...")
        print(f"{'━' * 50}")

        model = load_model(args.model_type, args.checkpoint, device)
        data = gather_eval_data(model, val_loader, NUM_CLASSES, device)

        tag = args.model_type
        print(f"\n📊 Generating plots for {model_name}...")
        plot_per_class_iou(data, model_name, os.path.join(args.output_dir, f"{tag}_per_class_iou.png"))
        plot_confusion_matrix(data, model_name, os.path.join(args.output_dir, f"{tag}_confusion_matrix.png"))
        plot_sample_predictions(data, model_name, os.path.join(args.output_dir, f"{tag}_sample_predictions.png"))
        plot_precision_recall_f1(data, model_name, os.path.join(args.output_dir, f"{tag}_precision_recall_f1.png"))
        plot_class_distribution(data, model_name, os.path.join(args.output_dir, f"{tag}_class_distribution.png"))
        plot_summary_dashboard(data, model_name, args.model_type, args.checkpoint,
                               os.path.join(args.output_dir, f"{tag}_dashboard.png"))

        # Print text summary too
        print(f"\n{'=' * 50}")
        print(f"Model       : {model_name}")
        print(f"mIoU        : {data['miou']:.4f}")
        print(f"Accuracy    : {data['accuracy']:.4f}")
        print(f"F1 (macro)  : {data['f1_macro']:.4f}")
        print(f"\nPer-Class IoU:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:<12}: {data['per_class_iou'][i]:.4f}")
        print(f"{'=' * 50}")

        # Save JSON
        summary = {k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in data.items() if k != "samples"}
        json_path = os.path.join(args.output_dir, f"{tag}_metrics.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved: {json_path}")

    print(f"\n✅ All plots saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
