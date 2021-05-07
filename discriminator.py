"""Discriminator models."""

import numpy as np
import torch.nn as nn
from layers import SNLinear
from layers import SNConv2d



class Discriminator(nn.Module):
    """Single expert.
    Args:
        args: argparse object
    """

    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.config = args
        self.model = self.build()

    def build(self):
        layers = [
            SNLinear(self.config.input_shape, 32),
            nn.LeakyReLU(),
            SNLinear(32, 64),
            nn.LeakyReLU(),
            SNLinear(64, self.config.discriminator_output_size),
        ]
        if self.config.discriminator_sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

    def forward(self, input):
        out = self.model(input)
        return out


class ConvolutionDiscriminator(Discriminator):
    """Convolutional discriminator."""

    def build(self):
        w = self.config.width_multiplier
        conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
        layers = [
            conv_layer(self.config.input_shape[-1], 16*w, 3, 2, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 16*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 16*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 32*w, 3, 2, padding=1),
            nn.LeakyReLU(),
            conv_layer(32*w, 32*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32*w, 64*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(self.config.input_shape[0]//4),
            nn.Flatten(),
            SNLinear(64*w, 100),
            nn.ReLU(),
            SNLinear(100, 1),
        ]
        if self.config.discriminator_sigmoid:
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)

class CifarConvolutionDiscriminator(nn.Module):
    """Convolutional discriminator."""

    def __init__(self):
        super(CifarConvolutionDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        self.ln1 = nn.LayerNorm(normalized_shape=[196, 32, 32])
        self.lrelu1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln2 = nn.LayerNorm(normalized_shape=[196, 16, 16])
        self.lrelu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln3 = nn.LayerNorm(normalized_shape=[196, 16, 16])
        self.lrelu3 = nn.LeakyReLU()

        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln4 = nn.LayerNorm(normalized_shape=[196, 8, 8])
        self.lrelu4 = nn.LeakyReLU()

        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln5 = nn.LayerNorm(normalized_shape=[196, 8, 8])
        self.lrelu5 = nn.LeakyReLU()

        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln6 = nn.LayerNorm(normalized_shape=[196, 8, 8])
        self.lrelu6 = nn.LeakyReLU()

        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.ln7 = nn.LayerNorm(normalized_shape=[196, 8, 8])
        self.lrelu7 = nn.LeakyReLU()

        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.ln8 = nn.LayerNorm(normalized_shape=[196, 4, 4])
        self.lrelu8 = nn.LeakyReLU()

        self.pool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.fc1 = nn.Linear(196, 1)

    def forward(self, x, print_size=False):
        if print_size:
            print("input size: {}".format(x.size()))

        x = self.conv1(x)
        x = self.ln1(x)
        x = self.lrelu1(x)

        if print_size:
            print(x.size())

        x = self.conv2(x)
        x = self.ln2(x)
        x = self.lrelu2(x)

        # the following lines are for extracting max feat from conv2 layer
        # h = F.max_pool2d(x, 4, 4)
        # h = h.view(-1, 196*4*4)
        # return h

        if print_size:
            print(x.size())

        x = self.conv3(x)
        x = self.ln3(x)
        x = self.lrelu3(x)

        if print_size:
            print(x.size())

        x = self.conv4(x)
        x = self.ln4(x)
        x = self.lrelu4(x)

        if print_size:
            print(x.size())

        x = self.conv5(x)
        x = self.ln5(x)
        x = self.lrelu5(x)

        if print_size:
            print(x.size())

        x = self.conv6(x)
        x = self.ln6(x)
        x = self.lrelu6(x)

        if print_size:
            print(x.size())

        x = self.conv7(x)
        x = self.ln7(x)
        x = self.lrelu7(x)

        if print_size:
            print(x.size())

        x = self.conv8(x)
        x = self.ln8(x)
        x = self.lrelu8(x)

        if print_size:
            print(x.size())

        x = self.pool(x)

        if print_size:
            print(x.size())

        x = x.view(x.size(0), -1)

        if print_size:
            print(x.size())

        fc1_out = self.fc1(x)

        return fc1_out


class MechanismConvolutionDiscriminator(nn.Module):

    def __init__(self, args):
        super(MechanismConvolutionDiscriminator, self).__init__()
        self.config = args
        self.model = self.build()

    def forward(self, input):
        out = self.model(input)
        return out

    def build(self):
        w = self.config.width_multiplier
        conv_layer = SNConv2d if self.config.use_sn else nn.Conv2d
        layers = [
            conv_layer(self.config.input_shape[-1], 16*w, 3, 2, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 16*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 16*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            conv_layer(16*w, 32*w, 3, 2, padding=1),
            nn.LeakyReLU(),
            conv_layer(32*w, 32*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32*w, 64*w, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.AvgPool2d(self.config.input_shape[0]//4),
            nn.Flatten(),
            SNLinear(64*w, 100),
            nn.ReLU(),
            SNLinear(100, self.config.num_experts),
        ]
        return nn.Sequential(*layers)
