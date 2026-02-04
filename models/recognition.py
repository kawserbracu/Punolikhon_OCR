from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
import math


class CRNN(nn.Module):
    """
    CRNN with VGG16_bn backbone (conv features) + BiLSTM + Linear to vocab.
    Output shape: (T, N, vocab_size) for CTC loss.
    """

    def __init__(self, vocab_size: int, pretrained_backbone: bool = True):
        super().__init__()
        vgg = vgg16_bn(pretrained=pretrained_backbone)
        # Use only convolutional features
        self.cnn = vgg.features  # (N, 512, H/32, W/32) nominally
        # Adjust downsampling to keep more width resolution: modify last maxpool to stride (2,1)
        # Find last MaxPool layers and tweak the last one
        for m in self.cnn.modules():
            if isinstance(m, nn.MaxPool2d) and m.kernel_size == (2, 2):
                last_pool = m
        # Avoid attribute error if not found
        try:
            last_pool.stride = (2, 1)
            last_pool.padding = (0, 0)
        except Exception:
            pass

        # Project features to a fixed channel size for RNN
        self.conv_proj = nn.Conv2d(512, 256, kernel_size=1)

        # Reduced BiLSTM (1 layer instead of 2, smaller hidden size)
        self.rnn = nn.LSTM(input_size=256 * 1, hidden_size=128, num_layers=1,
                           batch_first=False, bidirectional=True, dropout=0.2)
        
        # Simplified attention mechanism (fewer heads, smaller embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1, batch_first=False)
        self.attention_norm = nn.LayerNorm(256)
        
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, H, W)
        feats = self.cnn(x)          # (N, C=512, Hc, Wc)
        feats = self.conv_proj(feats)  # (N, 256, Hc, Wc)
        # Collapse height to 1 by averaging (simple, robust)
        feats = feats.mean(dim=2)    # (N, 256, Wc)
        # Prepare for LSTM: (Wc, N, 256)
        seq = feats.permute(2, 0, 1)
        out, _ = self.rnn(seq)       # (Wc, N, 256) - bidirectional 128*2=256
        
        # Apply self-attention for better context modeling
        attn_out, _ = self.attention(out, out, out)  # (Wc, N, 256)
        out = self.attention_norm(out + attn_out)  # Residual connection + LayerNorm
        
        logits = self.fc(out)        # (Wc, N, vocab)
        return logits

    @torch.no_grad()
    def ctc_decode(self, logits: torch.Tensor, blank_idx: int = 0) -> List[List[int]]:
        """
        Greedy decode per timestep: argmax, remove blanks and consecutive duplicates.
        logits: (T, N, V)
        returns: List of index sequences (length N)
        """
        probs = F.log_softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)  # (T, N)
        T, N = pred.shape
        results: List[List[int]] = []
        for n in range(N):
            seq = pred[:, n].tolist()
            decoded: List[int] = []
            prev = None
            for idx in seq:
                if idx == blank_idx:
                    prev = idx
                    continue
                if prev is not None and idx == prev:
                    prev = idx
                    continue
                decoded.append(idx)
                prev = idx
            results.append(decoded)
        return results
