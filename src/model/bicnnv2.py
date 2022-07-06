from torch import nn
import torch.nn.functional as F
from torch import cat


def make_model(args, parent=False):
    return BICNNV2(args)


class BICNNV2(nn.Module):
    def __init__(self, args):
        super(BICNNV2, self).__init__()
        num_channels = args.n_colors
        self.scale = args.scale[0]
        self.bconv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.bconv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.bconv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.fconv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.fconv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)  # TO-DO: the input channel of concat
        self.fconv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y=None):
        if y is None:
            x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
            fea = self.relu(self.fconv1(x))
            fea = self.relu(self.fconv2(fea.repeat(1, 2, 1, 1)))  # TO-DO: forward with single LR image
            # fea = self.relu(self.fconv2(fea))
            out = self.fconv3(fea)
            return out
        else:
            bfea1 = self.relu(self.bconv1(y))
            bfea2 = self.relu(self.bconv2(bfea1))
            outb = self.bconv3(bfea2)
            outb = F.interpolate(outb, scale_factor=1/self.scale, mode='bicubic')
            x = F.interpolate(x, scale_factor=self.scale, mode='bicubic')
            fea = self.relu(self.fconv1(x))
            fea = self.relu(self.fconv2(cat((fea, bfea1), dim=1)))  # TO-DO: concat back and forward hidden layer
            outf = self.fconv3(fea)
            return outf, outb
