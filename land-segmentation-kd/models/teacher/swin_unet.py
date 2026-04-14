"""
Hybrid Swin-UNet Teacher model for aerial land segmentation.

Architecture:
  - Encoder: Swin-Tiny (pretrained via timm, features_only)
  - Bottleneck: Dilated convolutions
  - Decoder: 4-level UNet with dilated convolutions + multi-scale fusion
  - Head: 1×1 conv → num_classes

Reference: Sundarr et al., "Enhanced aerial image segmentation via hybrid
Swin-UNet with dilated convolutions and multi-scale fusion", 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DilatedConvBlock(nn.Module):
    """Conv3×3 with dilation + BN + ReLU."""

    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MultiScaleFusion(nn.Module):
    """Fuse features from all encoder depths at a given decoder level.

    1. Project each encoder feature map to *out_ch* channels (1×1 conv).
    2. Resize all to *target_size* (bilinear).
    3. Concatenate and fuse with a 3×3 conv.
    """

    def __init__(self, channel_list, out_ch):
        super().__init__()
        self.adjusters = nn.ModuleList(
            [nn.Conv2d(c, out_ch, 1) for c in channel_list]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * len(channel_list), out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, features, target_size):
        aligned = []
        for feat, adj in zip(features, self.adjusters):
            f = F.interpolate(feat, size=target_size,
                              mode="bilinear", align_corners=False)
            aligned.append(adj(f))
        return self.fuse(torch.cat(aligned, dim=1))


def _to_bchw(t):
    """Ensure tensor is in (B, C, H, W) layout.
    
    timm's Swin-Tiny with features_only may return (B, H, W, C) on some
    versions. We detect this by checking if the last dimension looks like
    a channel dim (larger than spatial dims).
    """
    if t.dim() == 4:
        b, d1, d2, d3 = t.shape
        # If last dim is significantly larger than dim1 and dim2,
        # it's likely (B, H, W, C)
        if d3 > d1 and d3 > d2:
            return t.permute(0, 3, 1, 2).contiguous()
    return t


class SwinUNetTeacher(nn.Module):
    """Hybrid Swin-UNet with dilated decoder and multi-scale fusion.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes (default 6).
    pretrained : bool
        Load ImageNet-pretrained Swin-T backbone.
    """

    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()

        # ── Encoder (Swin-Tiny) ──────────────────────────────────────────
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        # Swin-Tiny channel dims at each stage
        enc_channels = [96, 192, 384, 768]

        # ── Bottleneck ───────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            DilatedConvBlock(enc_channels[3], enc_channels[3]),
            DilatedConvBlock(enc_channels[3], 512),
        )

        # ── Decoder blocks ───────────────────────────────────────────────
        self.decoder4 = DilatedConvBlock(512 + enc_channels[2], 256)
        self.decoder3 = DilatedConvBlock(256 + enc_channels[1], 128)
        self.decoder2 = DilatedConvBlock(128 + enc_channels[0], 64)
        self.decoder1 = DilatedConvBlock(64, 64)

        # ── Multi-scale fusion at each decoder level ─────────────────────
        self.msf4 = MultiScaleFusion(enc_channels, 256)
        self.msf3 = MultiScaleFusion(enc_channels, 128)
        self.msf2 = MultiScaleFusion(enc_channels, 64)

        # ── Segmentation head ────────────────────────────────────────────
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x, return_features=False):
        """Forward pass.

        Parameters
        ----------
        x : Tensor (B, 3, 224, 224)
        return_features : bool
            If True, also return decoder intermediate feature maps for
            knowledge distillation.

        Returns
        -------
        logits : Tensor (B, num_classes, 224, 224)
        features : dict[str, Tensor]  (only when return_features=True)
        """
        enc = self.encoder(x)
        f1, f2, f3, f4 = [_to_bchw(f) for f in enc]
        enc_bchw = [f1, f2, f3, f4]

        # Bottleneck
        x = self.bottleneck(f4)

        # Decoder stage 4
        x = F.interpolate(x, size=f3.shape[2:], mode="bilinear", align_corners=False)
        msf = self.msf4(enc_bchw, x.shape[2:])
        x = self.decoder4(torch.cat([x, f3], dim=1)) + msf
        d4 = x

        # Decoder stage 3
        x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=False)
        msf = self.msf3(enc_bchw, x.shape[2:])
        x = self.decoder3(torch.cat([x, f2], dim=1)) + msf
        d3 = x

        # Decoder stage 2
        x = F.interpolate(x, size=f1.shape[2:], mode="bilinear", align_corners=False)
        msf = self.msf2(enc_bchw, x.shape[2:])
        x = self.decoder2(torch.cat([x, f1], dim=1)) + msf
        d2 = x

        # Final upsample to full resolution (56 → 224 = 4×)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        x = self.decoder1(x)

        logits = self.head(x)

        if return_features:
            return logits, {"d4": d4, "d3": d3, "d2": d2}
        return logits


if __name__ == "__main__":
    # Quick sanity check
    model = SwinUNetTeacher(num_classes=6, pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    out, feats = model(dummy, return_features=True)
    print(f"Output shape : {out.shape}")
    for k, v in feats.items():
        print(f"  {k}: {v.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total / 1e6:.1f}M")
