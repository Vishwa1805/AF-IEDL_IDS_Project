"""Attention-enhanced IEDL-IDS model for the course-project ablation study.

The base CNN/MLP follows the project's IEDL-IDS v2 reconstruction. Optional
CBAM blocks refine the intermediate convolutional feature maps.
"""
from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.shared_mlp(self.avg_pool(x))
        attention = attention + self.shared_mlp(self.max_pool(x))
        return self.sigmoid(attention)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("Spatial attention kernel_size must be 3 or 7")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True).values
        return self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))


class CBAM(nn.Module):
    """Sequential channel and spatial attention (Woo et al., ECCV 2018)."""
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x) * x
        x = self.spatial(x) * x
        return x


class AttentionIEDLEncoder(nn.Module):
    def __init__(self, latent_dim: int = 20, use_cbam: bool = True) -> None:
        super().__init__()
        if not (1 <= latent_dim < 21):
            raise ValueError("latent_dim must be between 1 and 20 for n=21")
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.attn1 = CBAM(64) if use_cbam else nn.Identity()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.attn2 = CBAM(32) if use_cbam else nn.Identity()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.latent = nn.Linear(32 * 5 * 5, latent_dim)
        self.latent_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 21, 21):
            raise ValueError(f"Expected input (N,3,21,21), got {tuple(x.shape)}")
        x = self.pool1(self.attn1(self.relu1(self.conv1(x))))
        x = self.pool2(self.attn2(self.relu2(self.conv2(x))))
        x = self.flatten(x)
        if x.shape[1] != 800:
            raise RuntimeError(f"Expected 800 flattened features, got {x.shape[1]}")
        return self.latent_activation(self.latent(x))


class AttentionIEDLIDS(nn.Module):
    def __init__(self, num_classes: int = 9, latent_dim: int = 20,
                 hidden_dim: int = 256, dropout: float = 0.5,
                 use_cbam: bool = True) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.encoder = AttentionIEDLEncoder(latent_dim, use_cbam=use_cbam)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
