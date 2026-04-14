"""
Atrous Spatial Pyramid Pooling (ASPP) module.

Captures multi-scale context in the bottleneck by applying convolutions
with different dilation rates plus a global average pooling branch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPPConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling — captures multi-scale context in bottleneck"""
    def __init__(self, in_ch, out_ch=256, dilations=(6, 12, 18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)),  # 1×1
            *[ASPPConv(in_ch, out_ch, d) for d in dilations],              # dilated 3×3
            ASPPPooling(in_ch, out_ch),                                     # global avg pool
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(dilations) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.project(torch.cat([c(x) for c in self.convs], dim=1))
