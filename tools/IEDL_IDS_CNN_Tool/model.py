"""IEDL-IDS CNN encoder + MLP classifier (PyTorch).

Faithful components from the paper:
- RGB input image: 21x21x3
- Conv2d(3,64,3,padding=1) + ReLU + 2x2 max pool
- Conv2d(64,32,3,padding=1) + ReLU + 2x2 max pool
- Flatten: 32*5*5 = 800 values for 21x21 input
- Fully connected latent representation, followed by a neural classifier

The paper does not specify the exact latent width or every MLP layer dimension.
This implementation uses latent_dim=16 (< n=21) and one 256-unit hidden layer,
then a 9-class logits layer. These are documented implementation choices.
"""
from __future__ import annotations

import torch
from torch import nn


class IEDLEncoder(nn.Module):
    def __init__(self, latent_dim: int = 16) -> None:
        super().__init__()
        if not (1 <= latent_dim < 21):
            raise ValueError("latent_dim must be between 1 and 20 for n=21")
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.latent = nn.Linear(32 * 5 * 5, latent_dim)
        self.latent_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (3, 21, 21):
            raise ValueError(f"Expected NCHW input (N,3,21,21), got {tuple(x.shape)}")
        x = self.features(x)
        x = self.flatten(x)
        if x.shape[1] != 800:
            raise RuntimeError(f"Expected 800 flattened features, got {x.shape[1]}")
        return self.latent_activation(self.latent(x))


class IEDLIDS(nn.Module):
    """CNN encoder + MLP classifier; returns logits for CrossEntropyLoss."""
    def __init__(self, num_classes: int = 9, latent_dim: int = 16,
                 hidden_dim: int = 256, dropout: float = 0.5) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.encoder = IEDLEncoder(latent_dim=latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.classifier(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
