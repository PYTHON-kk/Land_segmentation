"""
Land segmentation inference - Input/Output prediction utility.

Generate segmentation predictions on input images using trained teacher or student model.

Usage (Single Image):
    python inference/predict_input_output.py --image path/to/image.jpg --model_type teacher
    python inference/predict_input_output.py --image path/to/image.jpg --model_type student

Usage (Batch Processing):
    python inference/predict_input_output.py --image_dir data/test_images --model_type teacher --output_dir predictions/

Usage (With Custom Checkpoint):
    python inference/predict_input_output.py --image image.jpg --checkpoint checkpoints/student_best.pth
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Set UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.mbrsc_dataset import CLASS_NAMES, NUM_CLASSES

# ─── Class Colors ────────────────────────────────────────────────────────────
CLASS_COLORS = np.array([
    [238,  76,  44],   # building  – coral red
    [196, 167, 125],   # land      – sandy tan
    [ 70,  70,  70],   # road      – dark gray
    [ 34, 197,  94],   # vegetation – green
    [ 56, 182, 255],   # water     – sky blue
    [160, 160, 160],   # unlabeled – light gray
], dtype=np.uint8)


# ─── Model Loading ───────────────────────────────────────────────────────────
def load_model(model_type, checkpoint, device):
    """Load model based on type."""
    if model_type == "teacher":
        try:
            from models.teacher.swin_unet_plus import SwinUNetPlusPlus
            model = SwinUNetPlusPlus(num_classes=6, pretrained=False, deep_supervision=True)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            if isinstance(state, dict) and 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ Teacher (Swin-Base UNet++) on {device} ({params / 1e6:.1f}M params)")
            return model.to(device).eval()
        except Exception as e:
            print(f"Falling back to old teacher: {e}")
            from models.teacher.swin_unet import SwinUNetTeacher
            model = SwinUNetTeacher(num_classes=6, pretrained=False)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ Teacher (old) on {device} ({params / 1e6:.1f}M params)")
            return model.to(device).eval()
    else:
        try:
            from models.student.efficientnet_unet import EfficientNetUNet
            model = EfficientNetUNet(num_classes=6, pretrained=False)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ Student (EfficientNet-B2) on {device} ({params / 1e6:.1f}M params)")
            return model.to(device).eval()
        except Exception as e:
            print(f"Falling back to old student: {e}")
            from models.student.mobilenet_unet import StudentUNet
            model = StudentUNet(num_classes=6, pretrained=False)
            state = torch.load(checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            params = sum(p.numel() for p in model.parameters())
            print(f"✓ Student (MobileNet) on {device} ({params / 1e6:.1f}M params)")
            return model.to(device).eval()


class LandSegmentationPredictor:
    """Aerial land segmentation predictor using teacher or student model."""

    def __init__(self, model_type, checkpoint, device="cuda"):
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.model = load_model(model_type, checkpoint, self.device)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def preprocess(self, img_bgr):
        """Load image, normalize, and convert to tensor."""
        # Ensure BGR to RGB
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Normalize
        img_float = img.astype(np.float32) / 255.0
        img_normalized = (img_float - self.mean) / self.std
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        return img_tensor, img.shape[:2]

    @torch.no_grad()
    def predict(self, img_bgr):
        """Generate prediction on input image.
        
        Returns:
            pred : ndarray (H, W) - class indices
            color_mask : ndarray (H, W, 3) - RGB colored mask
            probs : ndarray (H, W, 6) - class probabilities
            elapsed_ms : float - inference time in milliseconds
        """
        t0 = time.time()
        img_tensor, orig_shape = self.preprocess(img_bgr)
        
        logits = self.model(img_tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        elapsed_ms = (time.time() - t0) * 1000
        color_mask = CLASS_COLORS[pred]
        
        return pred, color_mask, probs, elapsed_ms


# ─── Visualization Utilities ─────────────────────────────────────────────────
def create_overlay(image, mask, alpha=0.6):
    """Create semi-transparent overlay of mask on original image."""
    mask_rgb = CLASS_COLORS[mask]
    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    return overlay


def save_visualization(img_bgr, pred_mask, output_path, image_name):
    """Create and save side-by-side visualization."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask_rgb = CLASS_COLORS[pred_mask]
    overlay = create_overlay(img_rgb, pred_mask, alpha=0.5)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    axes[1].imshow(mask_rgb)
    axes[1].set_title("Predicted Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (50%)", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    
    # Add legend
    patches = []
    for i in range(NUM_CLASSES):
        hex_color = f"#{CLASS_COLORS[i][0]:02x}{CLASS_COLORS[i][1]:02x}{CLASS_COLORS[i][2]:02x}"
        patches.append(mpatches.Patch(facecolor=hex_color, label=CLASS_NAMES[i]))
    
    fig.legend(handles=patches, loc="lower center", ncol=NUM_CLASSES,
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    
    plt.suptitle(f"Land Segmentation — {image_name}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_class_report(pred_mask):
    """Print per-class pixel distribution report."""
    unique, counts = np.unique(pred_mask, return_counts=True)
    total_pixels = pred_mask.size
    
    print("  Per-Class Distribution:")
    print("  " + "─" * 48)
    for c in range(NUM_CLASSES):
        if c in unique:
            count = counts[unique == c][0]
        else:
            count = 0
        pct = (count / total_pixels) * 100
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {CLASS_NAMES[c]:<12} {pct:>5.1f}% {bar}")
    print("  " + "─" * 48)


def main():
    parser = argparse.ArgumentParser(
        description="Land segmentation inference on input images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image prediction
  python inference/predict_input_output.py --image data/test.jpg --model_type teacher
  
  # Batch processing
  python inference/predict_input_output.py --image_dir data/test_images --model_type student
  
  # Custom checkpoint
  python inference/predict_input_output.py --image data/test.jpg --checkpoint checkpoints/teacher_best.pth
        """
    )
    
    parser.add_argument("--image", type=str, help="Path to single input image")
    parser.add_argument("--image_dir", type=str, help="Directory with multiple images")
    parser.add_argument("--model_type", choices=["teacher", "student"], 
                        default="teacher", help="Model architecture to use")
    parser.add_argument("--checkpoint", type=str, help="Custom checkpoint path")
    parser.add_argument("--output_dir", type=str, default="inference/outputs",
                        help="Output directory for predictions")
    parser.add_argument("--save_mask", action="store_true", default=True,
                        help="Save colored mask")
    parser.add_argument("--save_viz", action="store_true", default=True,
                        help="Save visualization (input + mask + overlay)")
    parser.add_argument("--device", type=str, default="cuda", 
                        choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Validate input
    if not args.image and not args.image_dir:
        parser.error("❌ Provide either --image or --image_dir")
    
    if args.image and not os.path.isfile(args.image):
        parser.error(f"❌ Image not found: {args.image}")
    
    if args.image_dir and not os.path.isdir(args.image_dir):
        parser.error(f"❌ Directory not found: {args.image_dir}")
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Set checkpoint
    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.model_type}_best.pth"
    
    if not os.path.isfile(args.checkpoint):
        parser.error(f"❌ Checkpoint not found: {args.checkpoint}")
    
    # Load model
    print(f"Loading {args.model_type} model...")
    predictor = LandSegmentationPredictor(args.model_type, args.checkpoint, device=args.device)
    print()
    
    # Collect image paths
    image_paths = []
    if args.image:
        image_paths = [args.image]
    else:
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if os.path.splitext(f)[1].lower() in valid_ext
        ]
        image_paths.sort()
    
    if not image_paths:
        print("❌ No images found")
        return
    
    print(f"Processing {len(image_paths)} image(s)...\n")
    
    # Process each image
    for idx, image_path in enumerate(image_paths, 1):
        image_name = Path(image_path).stem
        print(f"  [{idx}/{len(image_paths)}] {image_name}")
        
        try:
            # Read image
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                print(f"      ❌ Failed to read image")
                continue
            
            h, w = img_bgr.shape[:2]
            print(f"      Size: {w}×{h} pixels")
            
            # Predict
            pred_mask, color_mask, probs, elapsed_ms = predictor.predict(img_bgr)
            print(f"      Inference: {elapsed_ms:.1f}ms")
            
            # Print class distribution
            print_class_report(pred_mask)
            
            # Save outputs
            if args.save_mask:
                mask_path = os.path.join(args.output_dir, f"{image_name}_mask.png")
                color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
                cv2.imwrite(mask_path, color_mask_bgr)
                print(f"      ✓ Saved: {os.path.basename(mask_path)}")
            
            if args.save_viz:
                viz_path = os.path.join(args.output_dir, f"{image_name}_viz.png")
                save_visualization(img_bgr, pred_mask, viz_path, image_name)
                print(f"      ✓ Saved: {os.path.basename(viz_path)}")
            
            print()
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}\n")
            continue
    
    print(f"All predictions saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
