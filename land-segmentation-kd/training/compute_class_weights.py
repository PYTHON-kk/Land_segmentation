"""
Compute inverse-frequency class weights from training masks.

Fixes class imbalance (road/building are minority classes) by computing
per-class weights and saving to configs/class_weights.npy.

Usage:
    python training/compute_class_weights.py
"""

import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def compute_class_weights(mask_dir, split_file, num_classes=6):
    """
    Compute inverse-frequency class weights from training masks.
    Save result to configs/class_weights.npy
    """
    with open(split_file) as f:
        ids = [l.strip() for l in f.readlines() if l.strip()]

    counts = np.zeros(num_classes, dtype=np.float64)
    for name in tqdm(ids, desc="Counting pixels"):
        mask = np.load(Path(mask_dir) / f"{name}.npy")
        for c in range(num_classes):
            counts[c] += (mask == c).sum()

    # Inverse frequency, normalized
    weights = 1.0 / (counts + 1.0)
    weights = weights / weights.sum() * num_classes
    weights = np.clip(weights, 0.5, 5.0)  # prevent extreme weights

    os.makedirs("configs", exist_ok=True)
    np.save("configs/class_weights.npy", weights)
    class_names = ["building", "land", "road", "vegetation", "water", "unlabeled"]
    print("\nClass Weights:")
    for name, w, c in zip(class_names, weights, counts):
        print(f"  {name:<12}: weight={w:.3f}  pixels={int(c):,}")
    return weights


if __name__ == "__main__":
    compute_class_weights(
        mask_dir="data/processed/masks",
        split_file="data/splits/train.txt"
    )
