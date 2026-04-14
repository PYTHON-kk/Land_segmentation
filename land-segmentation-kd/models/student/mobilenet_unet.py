"""
Lightweight MobileNetV3-Small UNet student model for aerial land segmentation.

Features:
  - ~4M parameters (20× smaller than teacher)
  - ~10× faster inference than Swin-UNet
  - Projection heads to align intermediate feature channels
    with the teacher's decoder for feature distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBnRelu(nn.Module):
    """Conv3×3 + BN + ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StudentUNet(nn.Module):
    """MobileNetV3-Small encoder + lightweight UNet decoder.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    pretrained : bool
        Load ImageNet-pretrained MobileNetV3-Small backbone.
    """

    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────
        self.encoder = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )
        # MobileNetV3-Small channel dims at each extracted stage
        enc_channels = [16, 16, 24, 48, 576]

        # ── Decoder ──────────────────────────────────────────────────────
        self.decoder4 = ConvBnRelu(enc_channels[4] + enc_channels[3], 128)
        self.decoder3 = ConvBnRelu(128 + enc_channels[2], 64)
        self.decoder2 = ConvBnRelu(64 + enc_channels[1], 32)
        self.decoder1 = ConvBnRelu(32 + enc_channels[0], 16)

        # ── Segmentation head ────────────────────────────────────────────
        self.head = nn.Conv2d(16, num_classes, 1)

        # ── Projection heads (channel alignment for feature distillation)
        # These project student decoder features to match teacher dims:
        #   teacher d4 → 256ch,  teacher d3 → 128ch,  teacher d2 → 64ch
        self.proj_d4 = nn.Conv2d(128, 256, 1)
        self.proj_d3 = nn.Conv2d(64, 128, 1)
        self.proj_d2 = nn.Conv2d(32, 64, 1)

    def forward(self, x, return_features=False):
        """
        Parameters
        ----------
        x : Tensor (B, 3, 224, 224)
        return_features : bool
            If True, also return projected decoder feature maps.

        Returns
        -------
        logits : Tensor (B, num_classes, 224, 224)
        features : dict[str, Tensor]  (only when return_features=True)
        """
        enc = self.encoder(x)
        f0, f1, f2, f3, f4 = enc

        # Decoder
        d4 = self.decoder4(torch.cat([
            F.interpolate(f4, size=f3.shape[2:], mode="bilinear", align_corners=False),
            f3,
        ], dim=1))

        d3 = self.decoder3(torch.cat([
            F.interpolate(d4, size=f2.shape[2:], mode="bilinear", align_corners=False),
            f2,
        ], dim=1))

        d2 = self.decoder2(torch.cat([
            F.interpolate(d3, size=f1.shape[2:], mode="bilinear", align_corners=False),
            f1,
        ], dim=1))

        d1 = self.decoder1(torch.cat([
            F.interpolate(d2, size=f0.shape[2:], mode="bilinear", align_corners=False),
            f0,
        ], dim=1))

        # Upsample to full resolution
        logits = self.head(
            F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        )

        if return_features:
            return logits, {
                "d4": self.proj_d4(d4),
                "d3": self.proj_d3(d3),
                "d2": self.proj_d2(d2),
            }
        return logits


if __name__ == "__main__":
    # Quick sanity check
    model = StudentUNet(num_classes=6, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out, feats = model(dummy, return_features=True)
    print(f"Output shape : {out.shape}")
    for k, v in feats.items():
        print(f"  {k}: {v.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total / 1e6:.1f}M")
