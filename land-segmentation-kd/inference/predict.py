"""
Production inference with the distilled student model.

Loads the lightweight MobileNetV3-UNet, runs prediction on arbitrary
aerial images, and outputs a colour-coded segmentation mask.

Usage:
    python inference/predict.py --image sample.jpg
    python inference/predict.py --image sample.jpg --checkpoint checkpoints/student_best.pth
"""

import sys
import os
import time
import argparse

import torch
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.student.mobilenet_unet import StudentUNet

CLASS_COLORS = np.array([
    [60,  16,  152],   # building  — dark purple
    [132, 41,  246],   # land      — purple
    [110, 193, 228],   # road      — light blue
    [254, 221,  58],   # vegetation — yellow
    [226, 169,  41],   # water     — orange
    [155, 155, 155],   # unlabeled — grey
], dtype=np.uint8)

CLASS_NAMES = ["building", "land", "road", "vegetation", "water", "unlabeled"]


class LandSegmentationPredictor:
    """Fast aerial land segmentation predictor using the distilled student."""

    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model = StudentUNet(num_classes=6, pretrained=False)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        params = sum(p.numel() for p in self.model.parameters())
        print(f"Student model loaded on {self.device} ({params / 1e6:.1f}M params)")

    def preprocess(self, img_bgr):
        """Resize, normalise, and convert to tensor."""
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = (img / 255.0 - self.mean) / self.std
        return (
            torch.FloatTensor(img.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )

    def predict(self, img_bgr):
        """Run segmentation on a single image.

        Returns
        -------
        pred : ndarray (H, W)  — class indices
        color_mask : ndarray (H, W, 3) — RGB colour mask
        elapsed_ms : float — inference time in milliseconds
        """
        t0 = time.time()
        tensor = self.preprocess(img_bgr)
        with torch.no_grad():
            logits = self.model(tensor)
        pred = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        elapsed = (time.time() - t0) * 1000
        color_mask = CLASS_COLORS[pred]
        return pred, color_mask, elapsed

    def predict_tiled(self, img_bgr, tile_size=224):
        """Run tiled prediction on a large image.

        Splits the image into 224×224 tiles, predicts each, and
        stitches the result back together.
        """
        h, w = img_bgr.shape[:2]
        # Pad to multiples of tile_size
        pad_h = (tile_size - h % tile_size) % tile_size
        pad_w = (tile_size - w % tile_size) % tile_size
        padded = cv2.copyMakeBorder(img_bgr, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_REFLECT)
        ph, pw = padded.shape[:2]
        full_pred = np.zeros((ph, pw), dtype=np.uint8)

        t0 = time.time()
        for y in range(0, ph, tile_size):
            for x in range(0, pw, tile_size):
                tile = padded[y:y + tile_size, x:x + tile_size]
                pred, _, _ = self.predict(tile)
                full_pred[y:y + tile_size, x:x + tile_size] = pred
        elapsed = (time.time() - t0) * 1000

        # Remove padding
        full_pred = full_pred[:h, :w]
        color_mask = CLASS_COLORS[full_pred]
        return full_pred, color_mask, elapsed


def main():
    parser = argparse.ArgumentParser(description="Aerial land segmentation inference")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/student_best.pth")
    parser.add_argument("--output", type=str, default="predicted_mask.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tiled", action="store_true",
                        help="Use tiled prediction for large images")
    args = parser.parse_args()

    predictor = LandSegmentationPredictor(args.checkpoint, args.device)
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read {args.image}")
        return

    if args.tiled:
        pred, color_mask, ms = predictor.predict_tiled(img)
    else:
        pred, color_mask, ms = predictor.predict(img)

    print(f"Inference time: {ms:.1f} ms")
    print(f"Prediction shape: {pred.shape}")

    # Save colour mask
    cv2.imwrite(args.output, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    print(f"Saved prediction to {args.output}")

    # Print class distribution
    unique, counts = np.unique(pred, return_counts=True)
    total_px = pred.size
    print("\nClass distribution:")
    for cls_id, cnt in zip(unique, counts):
        pct = cnt / total_px * 100
        print(f"  {CLASS_NAMES[cls_id]:<12}: {pct:.1f}%")


if __name__ == "__main__":
    main()
