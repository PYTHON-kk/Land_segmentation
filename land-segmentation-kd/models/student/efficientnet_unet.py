"""
EfficientNet-B2 UNet student model for aerial land segmentation.

Features:
  - ~10M parameters (8× smaller than teacher)
  - ~2.5× faster inference than Swin-Base teacher
  - Projection heads for knowledge distillation feature alignment
  - Target after KD: ≥ 90% accuracy, ≥ 74% mIoU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetUNet(nn.Module):
    """
    EfficientNet-B2 encoder + lightweight UNet decoder
    ~10M params, ~2.5x faster than teacher
    Target after KD: ≥ 90% accuracy, ≥ 74% mIoU
    """
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            'efficientnet_b2',
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # skipping stem
        )
        # EfficientNet-B2 channel dims at indices 1-4: [24, 48, 120, 352]
        enc_ch = [24, 48, 120, 352]

        self.d3 = self._block(enc_ch[3] + enc_ch[2], 128)
        self.d2 = self._block(128 + enc_ch[1], 64)
        self.d1 = self._block(64  + enc_ch[0], 32)
        self.d0 = self._block(32, 32)

        self.head = nn.Conv2d(32, num_classes, 1)

        # Projection heads for knowledge distillation feature alignment
        self.proj_d3 = nn.Conv2d(128, 256, 1)   # align to teacher decoder dims
        self.proj_d2 = nn.Conv2d(64,  128, 1)
        self.proj_d1 = nn.Conv2d(32,   64, 1)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x, return_features=False):
        H, W = x.shape[2:]
        e1, e2, e3, e4 = self.encoder(x)

        d3 = self.d3(torch.cat([self._up(e4, e3), e3], dim=1))
        d2 = self.d2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.d1(torch.cat([self._up(d2, e1), e1], dim=1))
        d0 = self.d0(F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False))

        out = F.interpolate(self.head(d0), size=(H, W), mode='bilinear', align_corners=False)

        if return_features:
            return out, {
                "d3": self.proj_d3(d3),
                "d2": self.proj_d2(d2),
                "d1": self.proj_d1(d1),
            }
        return out


if __name__ == "__main__":
    # Quick sanity check
    model = EfficientNetUNet(num_classes=6, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out, feats = model(dummy, return_features=True)
    print(f"Output shape : {out.shape}")
    for k, v in feats.items():
        print(f"  {k}: {v.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total / 1e6:.1f}M")
