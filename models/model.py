"""
Fire Detection Models
- ModelA  : PyTorch CNN with same padding  (224->112->56->28->14, FC 6272->128)
- ModelB  : TF/Keras-equivalent CNN, valid padding (224->111->54->26->12, FC 4608->128->64)
- HybridFireDetector : Feature-level fusion of A + B (concat 192-d -> FC 96 -> FC 1)
"""

import torch
import torch.nn as nn


class ModelA(nn.Module):
    """
    PyTorch CNN with same (padding=1) convolutions.
    Spatial dims: 224 -> 112 -> 56 -> 28 -> 14
    Flattened: 14*14*32 = 6272  -->  FC(128)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Linear(32 * 14 * 14, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
        )
        self.head = nn.Linear(128, 1)

    def forward_features(self, x):
        return self.fc_block(self.flatten(self.features(x)))   # (B, 128)

    def forward(self, x):
        return torch.sigmoid(self.head(self.forward_features(x)))


class ModelB(nn.Module):
    """
    TF/Keras-equivalent CNN with valid (padding=0) convolutions.
    Spatial dims: 224 -> 111 -> 54 -> 26 -> 12
    Flattened: 12*12*32 = 4608  -->  FC(128) -> FC(64)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=0), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 3, padding=0), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=0), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Linear(4608, 128), nn.ReLU(inplace=True), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(inplace=True), nn.Dropout(0.4),
        )
        self.head = nn.Linear(64, 1)

    def forward_features(self, x):
        return self.fc_block(self.flatten(self.features(x)))   # (B, 64)

    def forward(self, x):
        return torch.sigmoid(self.head(self.forward_features(x)))


class HybridFireDetector(nn.Module):
    """
    Feature-level fusion:
        ModelA -> 128-d ─┐
                          concat(192-d) -> BN -> FC(96) -> FC(1) -> Sigmoid
        ModelB ->  64-d ─┘
    """

    def __init__(self):
        super().__init__()
        self.model_a = ModelA()
        self.model_b = ModelB()
        self.fusion_head = nn.Sequential(
            nn.BatchNorm1d(192),
            nn.Linear(192, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(96, 1),
        )

    def forward(self, x):
        fa = self.model_a.forward_features(x)   # (B, 128)
        fb = self.model_b.forward_features(x)   # (B,  64)
        return torch.sigmoid(self.fusion_head(torch.cat([fa, fb], dim=1)))

    def load_backbones(self, path_a: str, path_b: str, device):
        self.model_a.load_state_dict(torch.load(path_a, map_location=device))
        self.model_b.load_state_dict(torch.load(path_b, map_location=device))
        print("Backbone weights loaded successfully.")
