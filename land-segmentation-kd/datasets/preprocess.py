"""
Preprocessing script for the MBRSC Dubai Semantic Segmentation dataset.
Converts raw tiles into 224x224 patches with integer class masks.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# RGB class colors as they appear in the MBRSC mask PNGs
CLASS_COLORS = {
    "building":    [60,  16,  152],
    "land":        [132, 41,  246],
    "road":        [110, 193, 228],
    "vegetation":  [254, 221,  58],
    "water":       [226, 169,  41],
    "unlabeled":   [155, 155, 155],
}

CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_COLORS.keys())}


def color_mask_to_class(mask_rgb):
    """Convert an RGB mask to a single-channel integer class mask (H, W).
    
    Each pixel is assigned the class index whose RGB colour is closest
    (L2 distance) to the pixel's colour.  This handles anti-aliasing
    artefacts at class boundaries better than exact matching.
    """
    h, w = mask_rgb.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.uint8)

    # Build a colour palette array  (N_classes, 3)
    palette = np.array(list(CLASS_COLORS.values()), dtype=np.float32)
    flat = mask_rgb.reshape(-1, 3).astype(np.float32)

    # Compute squared distances to every class colour
    # shape: (H*W, N_classes)
    dists = np.sum((flat[:, None, :] - palette[None, :, :]) ** 2, axis=2)
    class_mask = np.argmin(dists, axis=1).astype(np.uint8).reshape(h, w)

    return class_mask


def tile_image(img, mask, tile_size=224, stride=224):
    """Tile image and mask into non-overlapping patches."""
    patches_img, patches_mask = [], []
    h, w = img.shape[:2]
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            patches_img.append(img[y:y + tile_size, x:x + tile_size])
            patches_mask.append(mask[y:y + tile_size, x:x + tile_size])
    return patches_img, patches_mask


def preprocess_dataset(raw_dir, output_dir, tile_size=224, resize_to=672):
    """Process all tiles: resize → convert mask → tile → save patches."""
    raw_dir = Path(raw_dir)
    img_out = Path(output_dir) / "images"
    mask_out = Path(output_dir) / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    idx = 0
    # The dataset is organised as Tile 1/ … Tile 8/ each with images/ and masks/
    tile_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("Tile")])
    print(f"Found {len(tile_dirs)} tile directories")

    for tile_dir in tile_dirs:
        img_dir = tile_dir / "images"
        msk_dir = tile_dir / "masks"
        if not img_dir.exists() or not msk_dir.exists():
            print(f"  Skipping {tile_dir.name}: missing images/ or masks/")
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            mask_path = msk_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                print(f"  Warning: no mask for {img_path.name}, skipping")
                continue

            img = cv2.imread(str(img_path))
            mask_bgr = cv2.imread(str(mask_path))
            if img is None or mask_bgr is None:
                print(f"  Warning: failed to read {img_path.name}, skipping")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

            # Resize to a size divisible by tile_size
            img_rgb = cv2.resize(img_rgb, (resize_to, resize_to))
            mask_rgb = cv2.resize(mask_rgb, (resize_to, resize_to),
                                  interpolation=cv2.INTER_NEAREST)

            # Convert RGB mask → integer class mask
            class_mask = color_mask_to_class(mask_rgb)

            # Tile
            img_patches, mask_patches = tile_image(img_rgb, class_mask, tile_size)
            for ip, mp in zip(img_patches, mask_patches):
                fname = f"patch_{idx:04d}"
                cv2.imwrite(str(img_out / f"{fname}.png"),
                            cv2.cvtColor(ip, cv2.COLOR_RGB2BGR))
                np.save(str(mask_out / f"{fname}.npy"), mp)
                idx += 1

    print(f"\nTotal patches created: {idx}")
    return idx


def create_splits(output_dir, num_patches, train_ratio=0.8, seed=42):
    """Create train / val split text files."""
    splits_dir = Path(output_dir).parent / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    ids = [f"patch_{i:04d}" for i in range(num_patches)]
    train_ids, val_ids = train_test_split(ids, train_size=train_ratio,
                                          random_state=seed, shuffle=True)

    for name, id_list in [("train", train_ids), ("val", val_ids)]:
        path = splits_dir / f"{name}.txt"
        with open(path, "w") as f:
            f.write("\n".join(id_list))
        print(f"  {name}: {len(id_list)} patches -> {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess MBRSC dataset")
    parser.add_argument("--raw_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), 
                                             "..", "Semantic segmentation dataset"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), 
                                             "..", "land-segmentation-kd", "data", "processed"))
    parser.add_argument("--tile_size", type=int, default=224)
    parser.add_argument("--resize_to", type=int, default=672)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    args = parser.parse_args()

    n = preprocess_dataset(args.raw_dir, args.output_dir,
                           args.tile_size, args.resize_to)
    create_splits(args.output_dir, n, args.train_ratio)
