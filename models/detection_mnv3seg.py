from __future__ import annotations
import torch
import torch.nn as nn


class LightBottleneck(nn.Module):
    """ResNet-style bottleneck with reduced channels."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        mid = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out + identity)
        return out


class MNV3Seg(nn.Module):
    """MobileNetV3 backbone + lightweight decoder to 1-channel segmentation map.
    Supports hybrid backbone with residual bottlenecks + multihead attention.
    """

    def __init__(self, backbone: str = 'large', pretrained: bool = True):
        super().__init__()
        from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
        self.backbone_name = backbone
        if backbone == 'large':
            self.backbone = mobilenet_v3_large(weights='DEFAULT' if pretrained else None).features
            ch = 960
            self.res_blocks = None
            self.attn = None
        elif backbone == 'hybrid':
            self.backbone = mobilenet_v3_small(weights='DEFAULT' if pretrained else None).features
            ch = 576
            self.res_blocks = nn.Sequential(
                LightBottleneck(ch, 256),
                LightBottleneck(256, 256),
                LightBottleneck(256, 128),
            )
            self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=False)
            ch = 128
        else:
            self.backbone = mobilenet_v3_small(weights='DEFAULT' if pretrained else None).features
            ch = 576
            self.res_blocks = None
            self.attn = None
        self.decoder = nn.Sequential(
            nn.Conv2d(ch, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 1, kernel_size=1),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.res_blocks is not None:
            feats = self.res_blocks(feats)
        if self.attn is not None:
            b, c, h, w = feats.shape
            seq = feats.view(b, c, h * w).permute(2, 0, 1)  # (HW,B,C)
            attn_out, _ = self.attn(seq, seq, seq, need_weights=False)
            attn_out = attn_out.permute(1, 2, 0).contiguous().view(b, c, h, w)
            feats = feats + attn_out  # skip connection
        out = self.decoder(feats)
        return out  # (N,1,H',W')
