"""
Swin-Base UNet++ Teacher model for aerial land segmentation.

Architecture:
  - Encoder: Swin-Base (pretrained via timm, features_only) — 4× capacity of Swin-Tiny
  - Bottleneck: ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context
  - Decoder: UNet++ dense skip connections for improved gradient flow
  - Deep Supervision: Auxiliary loss at each decoder level during training
  - Head: 1×1 conv → num_classes

Target: ≥ 95% accuracy, ≥ 82% mIoU on MBRSC dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .aspp import ASPP

# ── Helpers ──────────────────────────────────────────────────────────────────


def conv_bn_relu(in_ch, out_ch, k=3, d=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=d*(k//2), dilation=d, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def swin_to_spatial(x):
    """Handle both (B,H,W,C) and (B,C,H,W) outputs from timm Swin"""
    if x.dim() == 4 and x.shape[1] != x.shape[2] and x.shape[-1] < x.shape[-2]:
        return x  # already (B, C, H, W)
    if x.dim() == 4:
        return x.permute(0, 3, 1, 2).contiguous()  # (B,H,W,C) → (B,C,H,W)
    return x


# ── UNet++ Dense Decoder Block ────────────────────────────────────────────────


class DenseBlock(nn.Module):
    """UNet++ node: receives upsampled features + all same-level skip features"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv_bn_relu(in_ch, out_ch, d=1)
        self.conv2 = conv_bn_relu(out_ch, out_ch, d=2)   # dilated for wider RF
        self.conv3 = conv_bn_relu(out_ch, out_ch, d=1)
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x + r


# ── Main Teacher Model ────────────────────────────────────────────────────────


class SwinUNetPlusPlus(nn.Module):
    """
    Swin-Base backbone + UNet++ dense skip connections
    + ASPP bottleneck + deep supervision
    Target: ≥ 95% accuracy, ≥ 82% mIoU on MBRSC dataset
    """
    def __init__(self, num_classes=6, pretrained=True, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # ── Encoder: Swin-Base (4x larger capacity than Swin-Tiny) ──────────
        self.encoder = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        # Swin-Base channel dims: [128, 256, 512, 1024]
        C = [128, 256, 512, 1024]

        # ── Bottleneck: ASPP replaces single dilated conv ────────────────────
        self.bottleneck = ASPP(C[3], out_ch=512, dilations=(6, 12, 18))

        # ── UNet++ Decoder Nodes (i=decoder level, j=dense connection index) ─
        # Level 3 (1/8 spatial)
        self.node_3_0 = DenseBlock(C[2], 256)
        # Level 2 (1/4 spatial)
        self.node_2_0 = DenseBlock(C[1], 128)
        self.node_2_1 = DenseBlock(256 + 128 + C[1], 128)   # up(node_3_0) + node_2_0 + enc_2
        # Level 1 (1/2 spatial)
        self.node_1_0 = DenseBlock(C[0], 64)
        self.node_1_1 = DenseBlock(128 + 64 + C[0], 64)
        self.node_1_2 = DenseBlock(128 + 64 + 64 + C[0], 64)
        # Level 0 (full spatial)
        self.node_0_0 = DenseBlock(512 + C[3], 512)         # bot_up(512) + e3(1024) = 1536
        self.node_0_1 = DenseBlock(256 + 256, 256)           # align_bot(n0_0)→256 + n3_0(256) = 512
        self.node_0_2 = DenseBlock(128 + 128, 128)           # align_3(n0_1)→128 + n2_1(128) = 256
        self.node_0_3 = DenseBlock(64 + 64, 64)             # align_2(n0_2)→64 + n1_2(64) = 128

        # Channel alignment convolutions
        self.align_bot = nn.Conv2d(512, 256, 1)   # bottleneck → level3 size
        self.align_3   = nn.Conv2d(256, 128, 1)
        self.align_2   = nn.Conv2d(128, 64, 1)

        # ── Deep Supervision Heads (auxiliary outputs from each decoder level)
        if deep_supervision:
            self.aux_head3 = nn.Conv2d(256, num_classes, 1)
            self.aux_head2 = nn.Conv2d(128, num_classes, 1)
            self.aux_head1 = nn.Conv2d(64,  num_classes, 1)

        # ── Final head ───────────────────────────────────────────────────────
        self.final_conv = nn.Sequential(
            conv_bn_relu(64, 64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1),
        )

    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        H, W = x.shape[2:]

        # ── Encoder ──────────────────────────────────────────────────────────
        feats = self.encoder(x)
        e0 = swin_to_spatial(feats[0])  # (B,128,H/4,W/4)
        e1 = swin_to_spatial(feats[1])  # (B,256,H/8,W/8)
        e2 = swin_to_spatial(feats[2])  # (B,512,H/16,W/16)
        e3 = swin_to_spatial(feats[3])  # (B,1024,H/32,W/32)

        # ── Bottleneck ───────────────────────────────────────────────────────
        bot = self.bottleneck(e3)       # (B,512,H/32,W/32)

        # ── Dense UNet++ Decoder ─────────────────────────────────────────────
        # Level 3
        n3_0 = self.node_3_0(e2)
        bot_up = self._up(self.align_bot(bot), n3_0)

        # Level 2
        n2_0 = self.node_2_0(e1)
        n3_up = self._up(n3_0, n2_0)
        n2_1 = self.node_2_1(torch.cat([n3_up, n2_0, e1], dim=1))

        # Level 1
        n1_0 = self.node_1_0(e0)
        n2_up0 = self._up(n2_0, n1_0)
        n1_1 = self.node_1_1(torch.cat([n2_up0, n1_0, e0], dim=1))
        n2_up1 = self._up(n2_1, n1_0)
        n1_2 = self.node_1_2(torch.cat([n2_up1, n1_0, n1_1, e0], dim=1))

        # Level 0 (progressive upsampling to full res)
        n0_0 = self.node_0_0(torch.cat([self._up(bot, e3), e3], dim=1))
        n0_1 = self.node_0_1(torch.cat([self._up(self.align_bot(n0_0), n3_0), n3_0], dim=1))
        n0_2 = self.node_0_2(torch.cat([self._up(self.align_3(n0_1), n2_1), n2_1], dim=1))
        n0_3 = self.node_0_3(torch.cat([self._up(self.align_2(n0_2), n1_2), n1_2], dim=1))

        # Final upsample to input resolution
        out = F.interpolate(self.final_conv(n0_3), size=(H, W),
                            mode='bilinear', align_corners=False)

        if self.deep_supervision and self.training:
            aux3 = F.interpolate(self.aux_head3(n3_0), size=(H, W), mode='bilinear', align_corners=False)
            aux2 = F.interpolate(self.aux_head2(n2_1), size=(H, W), mode='bilinear', align_corners=False)
            aux1 = F.interpolate(self.aux_head1(n1_2), size=(H, W), mode='bilinear', align_corners=False)
            return out, aux3, aux2, aux1   # main + 3 auxiliary outputs

        return out


if __name__ == "__main__":
    # Quick sanity check
    model = SwinUNetPlusPlus(num_classes=6, pretrained=False, deep_supervision=True)
    dummy = torch.randn(2, 3, 224, 224)

    # Test training mode (deep supervision)
    model.train()
    outputs = model(dummy)
    print(f"Training mode outputs: {len(outputs)} tensors")
    for i, o in enumerate(outputs):
        print(f"  output[{i}]: {o.shape}")

    # Test eval mode (single output)
    model.eval()
    out = model(dummy)
    print(f"\nEval mode output: {out.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total / 1e6:.1f}M")
