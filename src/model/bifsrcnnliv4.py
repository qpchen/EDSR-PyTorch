from math import sqrt
from model import common

import torch
from torch import nn
import torch.nn.functional as F


def make_model(args, parent=False):
    return BIFSRCNNLIV4(args)


class BIFSRCNNLIV4(nn.Module):
    """
    Args:
        upscale_factor (int): Image magnification factor.
    """

    # def __init__(self, upscale_factor: int) -> None:
    def __init__(self, args):
        super(BIFSRCNNLIV4, self).__init__()

        upscale_factor = args.scale[0]
        num_channels = args.n_colors
        self.scale = args.scale[0]

        # RGB mean for DIV2K
        # self.sub_mean = common.MeanShift(args.rgb_range)
        # self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        ####################################################
        # Feature extraction layer.
        # backbone same as bifsrcnnv7
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, 56, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(56)
        )

        # Shrinking layer.
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Mapping layer.
        self.map1 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.map2 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.map3 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.map4 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Expanding layer.
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Deconvolution layer.
        # self.deconv = nn.ConvTranspose2d(56, num_channels, (9, 9), (upscale_factor, upscale_factor),
        #                                  (4, 4), (upscale_factor - 1, upscale_factor - 1))
        self.deconv = nn.Conv2d(56, num_channels, (5, 5), (1, 1), (2, 2))

        #################################################
        # Bi-direct training process layers
        # Bi-direct Deconv layer
        # self.bideconv = nn.Sequential(
        #     nn.Conv2d(num_channels, 56, (9, 9), (upscale_factor, upscale_factor), (4, 4)),
        #     nn.PReLU(56)
        # )
        self.bideconv = nn.Conv2d(num_channels, 56, (5, 5), (1, 1), (2, 2))

        # Bi-direct Expanding layer
        self.biexpand = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )

        # Bi-direct Mapping layer.
        # use same as self.map4-1
        self.bimap1 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.bimap2 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.bimap3 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.bimap4 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Bi-direct Shrinking layer
        self.bishrink = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )

        # Bi-direct Feature extraction layer
        self.bifeature = nn.Conv2d(56, num_channels, (5, 5), (1, 1), (2, 2))

        #################################################

        #################################################
        # Bi-direct loss process layers
        # Bi-direct Deconv layer
        # self.lsdeconv = nn.Sequential(
        #     nn.Conv2d(num_channels, num_channels, (3, 3), (1, 1), (1, 1)),
        #     nn.PReLU(num_channels)
        # )

        # Bi-direct Expanding layer
        # self.lsexpand = nn.Sequential(
        #     nn.Conv2d(56, 56, (3, 3), (1, 1), (1, 1)),
        #     nn.PReLU(56)
        # )

        # Bi-direct Mapping layer.
        self.lsmap1 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.lsmap2 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.lsmap3 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12))
        self.lsmap4 = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )
        self.lsmap = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Bi-direct Shrinking layer
        self.lsshrink = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )

        # Bi-direct Feature extraction layer
        # self.lsfeature = nn.Sequential(
        #     nn.Conv2d(56, 56, (3, 3), (1, 1), (1, 1)),
        #     nn.PReLU(56)
        # )
        #################################################

        # Initialize model weights.
        # self._initialize_weights()

    def forward(self, x, y=None):
        if y is None:
            # x = self.sub_mean(x)
            x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')

            fea1 = self.feature_extraction(x)
            fea2 = self.shrink(fea1)
            fea3 = self.map1(fea2)
            fea4 = self.map2(fea3)
            fea5 = self.map3(fea4)
            fea6 = self.map4(fea5)
            fea7 = self.expand(fea6)
            out = self.deconv(fea7)

            # out = self.add_mean(out)

            return out
        else:
            ############################
            # training from LR to HR
            x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
            fea1 = self.feature_extraction(x)
            fea2 = self.shrink(fea1)
            fea3 = self.map1(fea2)
            fea4 = self.map2(fea3)
            fea5 = self.map3(fea4)
            fea6 = self.map4(fea5)
            fea7 = self.expand(fea6)
            out = self.deconv(fea7)

            ############################
            # training from HR to LR
            bifea1 = self.bideconv(y)
            bifea2 = self.biexpand(bifea1)
            bifea3 = self.bimap4(bifea2)
            bifea4 = self.bimap3(bifea3)
            bifea5 = self.bimap2(bifea4)
            bifea6 = self.bimap1(bifea5)
            bifea7 = self.bishrink(bifea6)
            biout = self.bifeature(bifea7)
            biout = F.interpolate(biout, scale_factor=1 / self.scale, mode='bicubic')

            # return out, biout, fea2, fea3, fea4, fea5, fea6, bifea2, bifea3, bifea4, bifea5, bifea6
            return out, biout, self.lsshrink(fea2), self.lsmap1(fea3), self.lsmap2(fea4), self.lsmap3(fea5), self.lsmap4(fea6), bifea2, bifea3, bifea4, bifea5, bifea6

    # The filter weight of each layer is a Gaussian distribution with zero mean and
    # standard deviation initialized by random extraction 0.001 (deviation is 0).
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)

        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)
