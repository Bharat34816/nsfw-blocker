"""
NSFW Content Filter — Custom EfficientNet-B0 (From Scratch)

A faithful PyTorch implementation of the EfficientNet-B0 architecture
WITHOUT pre-trained weights. All layers are initialized from scratch
using Kaiming (He) initialization.

Architecture Overview:
    - Stem: Conv3x3 → BN → Swish
    - 7 stages of MBConv blocks (Mobile Inverted Bottleneck)
    - Each MBConv: Expansion → Depthwise Conv → Squeeze-and-Excitation → Projection
    - Head: Conv1x1 → GlobalAvgPool → Dropout → FC(2)
    - Total: ~5.3M parameters
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Building Blocks
# ===========================================================================

class Swish(nn.Module):
    """Swish activation: x * sigmoid(x)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block.

    Adaptively recalibrates channel-wise feature responses by
    modelling inter-channel dependencies.
    """

    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        squeezed = max(1, int(in_channels * se_ratio))
        self.fc1 = nn.Conv2d(in_channels, squeezed, kernel_size=1)
        self.fc2 = nn.Conv2d(squeezed, in_channels, kernel_size=1)
        self.swish = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.swish(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block (MBConv).

    Consists of:
        1. Expansion phase (1×1 conv, expand channels)
        2. Depthwise convolution (3×3 or 5×5)
        3. Squeeze-and-Excitation
        4. Projection phase (1×1 conv, reduce channels)
        5. Skip connection (if input/output shapes match)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0,
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_connect_rate = drop_connect_rate

        expanded = in_channels * expand_ratio
        layers = []

        # --- Expansion (skip if ratio == 1) ---
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded, momentum=0.01, eps=1e-3),
                Swish(),
            ])

        # --- Depthwise convolution ---
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(
                expanded, expanded, kernel_size,
                stride=stride, padding=padding,
                groups=expanded, bias=False,
            ),
            nn.BatchNorm2d(expanded, momentum=0.01, eps=1e-3),
            Swish(),
        ])

        # --- Squeeze-and-Excitation ---
        layers.append(SqueezeExcitation(expanded, se_ratio))

        # --- Projection ---
        layers.extend([
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)

        if self.use_residual:
            # Stochastic depth (drop connect) during training
            if self.training and self.drop_connect_rate > 0:
                keep_prob = 1.0 - self.drop_connect_rate
                mask = torch.rand(
                    (x.size(0), 1, 1, 1),
                    dtype=x.dtype, device=x.device,
                )
                mask = torch.floor(mask + keep_prob)
                out = out / keep_prob * mask

            out = out + x

        return out


# ===========================================================================
# EfficientNet-B0 Block Configuration
# ===========================================================================

@dataclass
class BlockConfig:
    """Configuration for a single MBConv stage."""
    num_repeats: int
    kernel_size: int
    stride: int
    expand_ratio: int
    in_channels: int
    out_channels: int
    se_ratio: float = 0.25


# EfficientNet-B0 configuration (from the original paper)
EFFICIENTNET_B0_CONFIG: List[BlockConfig] = [
    BlockConfig(1, 3, 1, 1, 32,  16,  0.25),   # Stage 1
    BlockConfig(2, 3, 2, 6, 16,  24,  0.25),   # Stage 2
    BlockConfig(2, 5, 2, 6, 24,  40,  0.25),   # Stage 3
    BlockConfig(3, 3, 2, 6, 40,  80,  0.25),   # Stage 4
    BlockConfig(3, 5, 1, 6, 80,  112, 0.25),   # Stage 5
    BlockConfig(4, 5, 2, 6, 112, 192, 0.25),   # Stage 6
    BlockConfig(1, 3, 1, 6, 192, 320, 0.25),   # Stage 7
]


# ===========================================================================
# EfficientNet-B0 Model
# ===========================================================================

class EfficientNetB0(nn.Module):
    """
    Custom EfficientNet-B0 for binary NSFW classification.

    Initialized entirely from scratch — no pre-trained weights.
    Uses Kaiming (He) initialization for convolutional layers
    and Xavier initialization for fully connected layers.

    Args:
        num_classes: Number of output classes (default: 2 for Safe/NSFW).
        dropout_rate: Dropout probability before the final FC layer.
        drop_connect_rate: Maximum drop-connect rate (linearly scaled per block).
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.drop_connect_rate = drop_connect_rate

        # --- Stem ---
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.01, eps=1e-3),
            Swish(),
        )

        # --- MBConv Stages ---
        self.blocks = self._build_blocks()

        # --- Head ---
        self.head_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280, momentum=0.01, eps=1e-3),
            Swish(),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1280, num_classes)

        # --- Initialize all weights from scratch ---
        self._initialize_weights()

    def _build_blocks(self) -> nn.Sequential:
        """Build all MBConv blocks according to B0 config."""
        blocks = []
        total_blocks = sum(cfg.num_repeats for cfg in EFFICIENTNET_B0_CONFIG)
        block_idx = 0

        for cfg in EFFICIENTNET_B0_CONFIG:
            for i in range(cfg.num_repeats):
                # First block in stage uses the configured stride;
                # subsequent blocks use stride=1 and in_channels=out_channels
                in_ch = cfg.in_channels if i == 0 else cfg.out_channels
                stride = cfg.stride if i == 0 else 1

                # Linearly scale drop-connect rate
                dc_rate = self.drop_connect_rate * block_idx / total_blocks

                blocks.append(
                    MBConvBlock(
                        in_channels=in_ch,
                        out_channels=cfg.out_channels,
                        kernel_size=cfg.kernel_size,
                        stride=stride,
                        expand_ratio=cfg.expand_ratio,
                        se_ratio=cfg.se_ratio,
                        drop_connect_rate=dc_rate,
                    )
                )
                block_idx += 1

        return nn.Sequential(*blocks)

    def _initialize_weights(self):
        """
        Initialize all weights from scratch.

        - Conv2d: Kaiming (He) normal initialization
        - BatchNorm2d: weight=1, bias=0
        - Linear: Xavier uniform initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Logits of shape (B, num_classes).
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


# ===========================================================================
# Factory & Utility
# ===========================================================================

def build_efficientnet_b0(num_classes: int = 2) -> EfficientNetB0:
    """Factory function to create an EfficientNet-B0 model."""
    model = EfficientNetB0(num_classes=num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EfficientNet-B0 created: {total_params:,} params ({trainable:,} trainable)")
    return model


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = build_efficientnet_b0(num_classes=2)
    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    proba = model.predict_proba(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Proba shape:  {proba.shape}")
    print(f"Probabilities: {proba}")
