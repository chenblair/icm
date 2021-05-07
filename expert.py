"""Definition of models."""

import numpy as np
import torch
import torch.nn as nn
from layers import SNLinear
from layers import SNConv2d


class Expert(nn.Module):
    """Single expert.

    Args:
        args: argparse object
    """

    def __init__(self, args):
        super(Expert, self).__init__()
        self.config = args
        self.model = self.build()

    def build(self):
        layers = [
            SNLinear(self.config.input_shape, 64),
            nn.ReLU(),
            SNLinear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.input_shape),
        ]
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.model(input)
        return out


class AffineExpert(Expert):
    """Expert that uses linear functions."""

    def build(self):
        layers = [
            nn.Linear(self.config.input_shape, self.config.input_shape),
        ]
        return nn.Sequential(*layers)


class TranslationExpert(AffineExpert):
    def build(self):
        layers = [
            nn.Linear(self.config.input_shape, self.config.input_shape),
        ]
        identity = np.float32([[1.0, 0.0], [0.0, 1.0]])
        layers[0].weight = nn.parameter.Parameter(
            torch.tensor(identity), requires_grad=False
        )
        return nn.Sequential(*layers)


class ConvolutionExpert(Expert):
    """Expert for images."""

    def build(self):
        conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
        layers = [
            conv_layer(self.config.input_shape[-1], 16, 3, 1, padding=1),
            nn.ReLU(),
            conv_layer(16, 32, 3, 1, padding=1),
            nn.ReLU(),
            conv_layer(32, 16, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1, 1, padding=0),
            nn.Sigmoid(),
        ]
        return nn.Sequential(*layers)

class CifarConvolutionExpert(Expert):
    """Expert for images."""

    def build(self):
        conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
        layers = [
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            nn.ReLU(),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)