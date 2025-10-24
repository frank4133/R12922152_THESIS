import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch

def crop_to_match(tensor, target_tensor):
    """
    Crop the input tensor to match the size of the target tensor.
    """
    _, _, h, w = target_tensor.size()
    return tensor[:, :, :h, :w]

class Res_block(nn.Module):
    def __init__(self, nFeat, kernel_size=3):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.conv2(out) + x
        return out

class Encoder(nn.Module):
    def __init__(self, in_channels=3, mid_channels=32, out_channels=64, down_level=3):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride = 2, padding=1)
        self.res_block1 = Res_block(mid_channels)
        self.activation = nn.GELU()
        self.down = nn.ModuleList()  # Use nn.ModuleList to hold the modules
        # first level: 32 -> 64
        # second level: 64 -> 128
        # third level: 128 -> 256
        # fourth level: 256 -> 512
        for i in range(down_level):
            modules = nn.Sequential(
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1),
                Res_block(out_channels)
            )
            self.down.append(modules)
            mid_channels *= 2
            out_channels *= 2

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.res_block1(x)
        for i in range(len(self.down)):
            x = self.down[i](x)
        return x

class DDMEF(nn.Module):
    def __init__(self, opt):
        super(DDMEF, self).__init__()
        filters_in = 3
        filters_out = 3
        nFeat = 32

        # encoder1
        self.conv1 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv1_E0_1 = nn.Conv2d(nFeat, nFeat, 3, 2, 1, bias=True)
        self.conv1_E0_2 = Res_block(nFeat)
        self.conv1_E1_1 = nn.Conv2d(nFeat, nFeat * 2, 3, 2, 1, bias=True)
        self.conv1_E1_2 = Res_block(nFeat * 2)
        self.conv1_E2_1 = nn.Conv2d(nFeat * 2, nFeat * 4, 3, 2, 1, bias=True)
        self.conv1_E2_2 = Res_block(nFeat * 4)
        self.conv1_E3_1 = nn.Conv2d(nFeat * 4, nFeat * 8, 3, 2, 1, bias=True)
        self.conv1_E3_2 = Res_block(nFeat * 8)

        # encoder2
        self.conv2 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv2_E0_1 = nn.Conv2d(nFeat, nFeat, 3, 2, 1, bias=True)
        self.conv2_E0_2 = Res_block(nFeat)
        self.conv2_E1_1 = nn.Conv2d(nFeat, nFeat * 2, 3, 2, 1, bias=True)
        self.conv2_E1_2 = Res_block(nFeat * 2)
        self.conv2_E2_1 = nn.Conv2d(nFeat * 2, nFeat * 4, 3, 2, 1, bias=True)
        self.conv2_E2_2 = Res_block(nFeat * 4)
        self.conv2_E3_1 = nn.Conv2d(nFeat * 4, nFeat * 8, 3, 2, 1, bias=True)
        self.conv2_E3_2 = Res_block(nFeat * 8)

        # merge
        self.merger = nn.Conv2d(nFeat * 16, nFeat * 8, 3, 1, 1, bias=True)
        self.res_module = Res_block(nFeat * 8)

        # decoder
        self.conv_D3_1 = nn.ConvTranspose2d(nFeat * 8, nFeat * 4, 4, 2, 1, bias=True)
        self.conv_D3_2 = Res_block(nFeat * 4)
        self.conv_D2_1 = nn.ConvTranspose2d(nFeat * 12, nFeat * 2, 4, 2, 1, bias=True)
        self.conv_D2_2 = Res_block(nFeat * 2)
        self.conv_D1_1 = nn.ConvTranspose2d(nFeat * 6, nFeat, 4, 2, 1, bias=True)
        self.conv_D1_2 = Res_block(nFeat)
        self.conv_D0_1 = nn.ConvTranspose2d(nFeat * 3, nFeat, 4, 2, 1, bias=True)
        self.conv_D0_2 = Res_block(nFeat)
        self.conv_D_1 = nn.Conv2d(nFeat * 3, nFeat * 3, 3, 1, 1, bias=True)
        self.conv_D_2 = Res_block(nFeat * 3)

        self.conv_out = nn.Conv2d(nFeat * 3, 3, 3, 1, 1, bias=True)
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):

         # Encoder1
        E1 = self.activation(self.conv1(x1))
        # print(f"E1: {E1.shape}")
        E1_0 = self.activation(self.conv1_E0_1(E1))
        E1_0 = self.conv1_E0_2(E1_0)
        # print(f"E1_0: {E1_0.shape}")
        E1_1 = self.activation(self.conv1_E1_1(E1_0))
        E1_1 = self.conv1_E1_2(E1_1)
        # print(f"E1_1: {E1_1.shape}")
        E1_2 = self.activation(self.conv1_E2_1(E1_1))
        E1_2 = self.conv1_E2_2(E1_2)
        # print(f"E1_2: {E1_2.shape}")
        E1_3 = self.activation(self.conv1_E3_1(E1_2))
        E1_3 = self.conv1_E3_2(E1_3)
        # print(f"E1_3: {E1_3.shape}")

        # Encoder2
        E2 = self.activation(self.conv2(x2))
        E2_0 = self.activation(self.conv2_E0_1(E2))
        E2_0 = self.conv2_E0_2(E2_0)
        E2_1 = self.activation(self.conv2_E1_1(E2_0))
        E2_1 = self.conv2_E1_2(E2_1)
        E2_2 = self.activation(self.conv2_E2_1(E2_1))
        E2_2 = self.conv2_E2_2(E2_2)
        E2_3 = self.activation(self.conv2_E3_1(E2_2))
        E2_3 = self.conv2_E3_2(E2_3)

        feat_merged = self.activation(self.merger(torch.cat([E1_3, E2_3], dim = 1)))
        res_tensor = self.res_module(feat_merged)
        # print(f"res_tensor: {res_tensor.shape}")

        # Decoder
        D3 = self.activation(self.conv_D3_1(res_tensor))
        D3 = self.conv_D3_2(D3)
        # print(f"D3: {D3.shape}")
        D2 = self.activation(self.conv_D2_1(torch.cat([D3, E1_2, E2_2], dim=1)))
        D2 = self.conv_D2_2(D2)
        # print(f"D2: {D2.shape}")
        D1 = self.activation(self.conv_D1_1(torch.cat([D2, E1_1, E2_1], dim=1)))
        D1 = self.conv_D1_2(D1)
        # print(f"D1: {D1.shape}")
        D0 = self.activation(self.conv_D0_1(torch.cat([D1, E1_0, E2_0], dim=1)))
        D0 = self.conv_D0_2(D0)
        # print(f"D0: {D0.shape}")
        D = self.activation(self.conv_D_1(torch.cat([D0, E1, E2], dim=1)))
        D = self.conv_D_2(D)
        # print(f"D: {D.shape}")

        # print(f"output: {output.shape}")
        out = self.sigmoid(self.conv_out(D))
        #return D3, D2, D1, D0, D, out # for training
        return out


