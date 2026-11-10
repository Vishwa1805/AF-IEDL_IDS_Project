"""IEDL-IDS CNN encoder + MLP classifier (paper-guided reconstruction).

Paper-specified encoder for 21x21x3 ToN_IoT images:
  Conv2d(3,64,3,padding=1) -> ReLU -> MaxPool2d(2)
  Conv2d(64,32,3,padding=1) -> ReLU -> MaxPool2d(2)
  Flatten 32*5*5 = 800 -> fully-connected latent layer.

The paper requires latent width k < n (n=21), but does not publish k.
This version uses k=20. The paper's MLP description specifies 256-node dense
layers with dropout 0.5; this version uses two 256-unit dense layers (dropout
after the first) followed by a 9-class logits layer. Exact latent/MLP details
remain documented reconstruction choices, not confirmed author code.
"""
from __future__ import annotations
import torch
from torch import nn

class IEDLEncoder(nn.Module):
    def __init__(self, latent_dim: int = 20) -> None:
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
        x = self.flatten(self.features(x))
        if x.shape[1] != 800:
            raise RuntimeError(f"Expected 800 flattened features, got {x.shape[1]}")
        return self.latent_activation(self.latent(x))

class IEDLIDS(nn.Module):
    """CNN encoder + two-layer MLP classifier; returns class logits."""
    def __init__(self, num_classes: int = 9, latent_dim: int = 20,
                 hidden_dim: int = 256, dropout: float = 0.5) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        self.encoder = IEDLEncoder(latent_dim=latent_dim)
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
