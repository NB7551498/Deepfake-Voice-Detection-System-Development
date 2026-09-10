"""
Deepfake Fusion Model 2.0.

Combines 228 classical acoustic features with 768-dimensional frozen SSL embeddings
into a single 996-dimensional representation, classified via a small fusion head.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class DeepfakeFusion2_0(nn.Module):
    """
    Lightweight 996-dim Fusion Classifier.
    - 228 acoustic features (interpretable vocal tract & harmonic envelope)
    - 768 frozen SSL embeddings (temporal speech representation)
    Total parameters: ~270K (< 1.2 MB disk space).
    """
    def __init__(self, acoustic_dim: int = 228, ssl_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.input_dim = acoustic_dim + ssl_dim

        # Acoustic pathway projection
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # SSL pathway projection
        self.ssl_proj = nn.Sequential(
            nn.Linear(ssl_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )

        # Joint fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, acoustic_feats: torch.Tensor, ssl_feats: torch.Tensor) -> torch.Tensor:
        """
        acoustic_feats: (B, 228)
        ssl_feats: (B, 768)
        Returns: (B, 1) calibrated fake probability in [0, 1]
        """
        h_ac = self.acoustic_proj(acoustic_feats)
        h_ssl = self.ssl_proj(ssl_feats)
        fused = torch.cat([h_ac, h_ssl], dim=-1)
        return self.classifier(fused)
