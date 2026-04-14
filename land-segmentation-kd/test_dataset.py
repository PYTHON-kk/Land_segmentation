"""Quick test to verify dataset loading."""
from datasets.mbrsc_dataset import MBRSCDataset, get_train_transforms
from torch.utils.data import DataLoader

ds = MBRSCDataset("data/processed/images", "data/processed/masks",
                   "data/splits/train.txt", get_train_transforms())
print(f"Dataset size: {len(ds)}")
img, mask = ds[0]
print(f"Image shape: {img.shape}, dtype: {img.dtype}")
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
print(f"Mask classes: {mask.unique().tolist()}")

loader = DataLoader(ds, batch_size=4, shuffle=True)
for imgs, masks in loader:
    print(f"Batch images: {imgs.shape}, masks: {masks.shape}")
    break
print("Dataset OK!")
