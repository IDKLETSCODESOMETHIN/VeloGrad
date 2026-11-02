# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# PRODUCTION MODEL: ResNet Vision Architecture
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad Native ResNet Vision Architecture.
Residual convolutional network with identity shortcuts.
"""

from typing import List, Optional
from .. import Tensor, nn, functional as F

class BasicBlock(nn.Module):
    """Residual building block for ResNet."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
        else:
            self.shortcut = nn.Sequential()

        self._submodules = [self.conv1, self.conv2, self.shortcut]

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)

class ResNet18(nn.Module):
    """ResNet-18 model architecture."""
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU()
        self.layer1 = BasicBlock(64, 64)
        self.layer2 = BasicBlock(64, 128, stride=2)
        self.layer3 = BasicBlock(128, 256, stride=2)
        self.layer4 = BasicBlock(256, 512, stride=2)
        self.fc = nn.Linear(512, num_classes)
        self._submodules = [self.conv1, self.layer1, self.layer2, self.layer3, self.layer4, self.fc]

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return self.fc(out)
