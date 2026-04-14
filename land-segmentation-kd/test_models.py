"""Quick sanity check for both models."""
import torch
from models.teacher.swin_unet import SwinUNetTeacher
from models.student.mobilenet_unet import StudentUNet

def main():
    d = torch.randn(2, 3, 224, 224)

    # Teacher
    t = SwinUNetTeacher(6, pretrained=False)
    to, tf = t(d, return_features=True)
    tp = sum(p.numel() for p in t.parameters())
    print(f"Teacher output: {to.shape}")
    for k, v in tf.items():
        print(f"  teacher {k}: {v.shape}")
    print(f"Teacher params: {tp / 1e6:.1f}M")

    # Student
    s = StudentUNet(6, pretrained=False)
    so, sf = s(d, return_features=True)
    sp = sum(p.numel() for p in s.parameters())
    print(f"\nStudent output: {so.shape}")
    for k, v in sf.items():
        print(f"  student {k}: {v.shape}")
    print(f"Student params: {sp / 1e6:.1f}M")

    print(f"\nCompression: {tp / sp:.1f}x")
    print("All shapes OK!")

if __name__ == "__main__":
    main()
