import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


class STN(nn.Module):
    """Spatial Transformer Network to deskew/normalize handwriting."""

    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(True),
            nn.Linear(128, 6),
        )
        # Initialize as identity transform
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x)
        # Adaptively pool to fixed spatial size to avoid shape mismatch on varying widths/heights
        xs = F.adaptive_avg_pool2d(xs, (20, 20))
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


class MobileNetV3_FeatureExtractor(nn.Module):
    """MobileNetV3-Small backbone without classifier."""

    def __init__(self, weights: str = "DEFAULT"):
        super().__init__()
        model = mobilenet_v3_small(weights=weights)
        self.features = model.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


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


class HybridCNN(nn.Module):
    """MobileNetV3-Small + lightweight bottlenecks."""

    def __init__(self, backbone_weights: str = "DEFAULT"):
        super().__init__()
        self.backbone = MobileNetV3_FeatureExtractor(weights=backbone_weights)
        self.res_blocks = nn.Sequential(
            LightBottleneck(576, 256),
            LightBottleneck(256, 256),
            LightBottleneck(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.res_blocks(x)
        return x


def cnn_to_sequence(x: torch.Tensor) -> torch.Tensor:
    # (B,C,H,W) -> (W,B,C) for CTC
    x = x.mean(dim=2)  # collapse height
    x = x.permute(2, 0, 1)  # width-major
    return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len,1,d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,B,C)
        return x + self.pe[: x.size(0)]


class TransformerSequence(nn.Module):
    def __init__(self, input_dim: int, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T,B,C)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x


class HybridHTR_STN_Transformer(nn.Module):
    """STN + MobileNetV3-Small + bottlenecks + Transformer encoder (CTC-ready)."""

    def __init__(self, num_classes: int, backbone_weights: str = "DEFAULT"):
        super().__init__()
        self.stn = STN()
        self.cnn = HybridCNN(backbone_weights=backbone_weights)
        self.sequence = TransformerSequence(input_dim=128, nhead=4, num_layers=3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stn(x)
        x = self.cnn(x)
        x = cnn_to_sequence(x)  # (T,B,C)
        x = self.sequence(x)
        x = self.classifier(x)
        return x  # (T,B,V)


class HybridHTR_STN_Transformer_BiLSTM(nn.Module):
    """Hybrid STN+CNN with a single BiLSTM before the Transformer encoder."""

    def __init__(self, num_classes: int, backbone_weights: str = "DEFAULT"):
        super().__init__()
        self.stn = STN()
        self.cnn = HybridCNN(backbone_weights=backbone_weights)
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128 // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=False,
        )
        self.sequence = TransformerSequence(input_dim=128, nhead=4, num_layers=3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stn(x)
        x = self.cnn(x)
        x = cnn_to_sequence(x)  # (T,B,C)
        x, _ = self.bilstm(x)  # (T,B,C)
        x = self.sequence(x)
        x = self.classifier(x)
        return x  # (T,B,V)
