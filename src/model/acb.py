import torch.nn as nn
import torch.nn.init as init
import torch

class LayerNorm(nn.Module):
    r""" LayerNorm that supports input data with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_affine=True, reduce_gamma=False, gamma_init=None, norm='batch'):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        self.norm = norm
        if self.norm not in ["batch", "layer", "no", "v8old"]:
            raise NotImplementedError 
        if self.norm == "v8old":  # the v8old do not training bias, so fused conv should not init bias. This is not recommend!!!
            fused_bias = False
        else:
            fused_bias = True
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=fused_bias, padding_mode=padding_mode)
        else:
            if self.norm == "batch":
                self.square_n = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
                self.conv_bias = False
            elif self.norm == "layer":
                self.square_n = LayerNorm(out_channels)
                self.conv_bias = False
            elif self.norm == "no":
                self.conv_bias = True
            elif self.norm == "v8old":
                self.conv_bias = False
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=self.conv_bias,
                                         padding_mode=padding_mode)


            if padding - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding, dilation=dilation, groups=groups, bias=self.conv_bias,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding, dilation=dilation, groups=groups, bias=self.conv_bias,
                                      padding_mode=padding_mode)
            if self.norm == "batch":
                self.ver_n = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
                self.hor_n = nn.BatchNorm2d(num_features=out_channels, affine=use_affine)
            elif self.norm == "layer":
                self.ver_n = LayerNorm(out_channels)
                self.hor_n = LayerNorm(out_channels)

                if reduce_gamma:
                    self.init_gamma(1.0 / 3)

                if gamma_init is not None:
                    assert not reduce_gamma
                    self.init_gamma(gamma_init)

    # ############## when use batch norm #################
    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _fuse_ln_tensor(self, conv, bn):
        # TODO: change formula to ln
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        if self.norm == "batch":
            hor_k, hor_b = self._fuse_bn_tensor(self.hor_conv, self.hor_n)
            ver_k, ver_b = self._fuse_bn_tensor(self.ver_conv, self.ver_n)
            square_k, square_b = self._fuse_bn_tensor(self.square_conv, self.square_n)
        elif self.norm == "layer":
            hor_k, hor_b = self._fuse_ln_tensor(self.hor_conv, self.hor_n)
            ver_k, ver_b = self._fuse_ln_tensor(self.ver_conv, self.ver_n)
            square_k, square_b = self._fuse_ln_tensor(self.square_conv, self.square_n)
        elif self.norm == "no":
            square_k = self.square_conv.weight.detach()
            square_b = self.square_conv.bias
            hor_k = self.hor_conv.weight
            hor_b = self.hor_conv.bias
            ver_k = self.ver_conv.weight
            ver_b = self.ver_conv.bias
        elif self.norm == "v8old":  # should be deleted
            square_k = self.square_conv.weight.detach()
            hor_k = self.hor_conv.weight
            ver_k = self.ver_conv.weight
            # fuse_b = self.square_conv.bias  # should be NoneType since bias set to False
        self._add_to_square_kernel(square_k, hor_k)
        self._add_to_square_kernel(square_k, ver_k)
        if self.conv_bias:
            return square_k, hor_b + ver_b + square_b
        else:
            return square_k
    # ###################################################


    def switch_to_deploy(self):
        if self.conv_bias:
            deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        else:
            deploy_k = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.fused_conv = nn.Conv2d(in_channels=self.square_conv.in_channels, out_channels=self.square_conv.out_channels,
                                    kernel_size=self.square_conv.kernel_size, stride=self.square_conv.stride,
                                    padding=self.square_conv.padding, dilation=self.square_conv.dilation, groups=self.square_conv.groups, bias=self.conv_bias,
                                    padding_mode=self.square_conv.padding_mode)
        self.__delattr__('square_conv')
        self.__delattr__('hor_conv')
        self.__delattr__('ver_conv')
        if self.norm == "batch" or self.norm == "layer":
            self.__delattr__('square_n')
            self.__delattr__('hor_n')
            self.__delattr__('ver_n')
        self.fused_conv.weight.data = deploy_k
        if self.conv_bias:
            self.fused_conv.bias.data = deploy_b


    def init_gamma(self, gamma_value):
        init.constant_(self.square_n.weight, gamma_value)
        init.constant_(self.ver_n.weight, gamma_value)
        init.constant_(self.hor_n.weight, gamma_value)
        print('init gamma of square, ver and hor as ', gamma_value)

    def single_init(self):
        init.constant_(self.square_n.weight, 1.0)
        init.constant_(self.ver_n.weight, 0.0)
        init.constant_(self.hor_n.weight, 0.0)
        print('init gamma of square as 1, ver and hor as 0')

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            if self.norm == "batch" or self.norm == "layer":
                square_outputs = self.square_n(square_outputs)
            if self.crop > 0:
                ver_input = input[:, :, :, self.crop:-self.crop]
                hor_input = input[:, :, self.crop:-self.crop, :]
            else:
                ver_input = input
                hor_input = input
            vertical_outputs = self.ver_conv(ver_input)
            if self.norm == "batch" or self.norm == "layer":
                vertical_outputs = self.ver_n(vertical_outputs)
            horizontal_outputs = self.hor_conv(hor_input)
            if self.norm == "batch" or self.norm == "layer":
                horizontal_outputs = self.hor_n(horizontal_outputs)
            result = square_outputs + vertical_outputs + horizontal_outputs
            return result

if __name__ == '__main__':
    N = 1
    C = 2
    H = 62
    W = 62
    O = 8
    groups = 4

    x = torch.randn(N, C, H, W)
    print('input shape is ', x.size())

    test_kernel_padding = [(3,1), (3,0), (5,1), (5,2), (5,3), (5,4), (5,6)]

    for k, p in test_kernel_padding:
        acb = ACBlock(C, O, kernel_size=k, padding=p, stride=1, deploy=False)
        acb.eval()
        for module in acb.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.uniform_(module.running_mean, 0, 0.1)
                nn.init.uniform_(module.running_var, 0, 0.2)
                nn.init.uniform_(module.weight, 0, 0.3)
                nn.init.uniform_(module.bias, 0, 0.4)
        out = acb(x)
        acb.switch_to_deploy()
        deployout = acb(x)
        print('difference between the outputs of the training-time and converted ACB is')
        print(((deployout - out) ** 2).sum())

