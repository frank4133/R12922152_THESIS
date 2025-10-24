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

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, ref_img):
        x = torch.cat([img, ref_img], dim=1)
        out = self.activation(self.conv1(x))
        out = self.sigmoid(self.conv2(out))
        return out * img

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

class SUMEF(nn.Module):
    def __init__(self, opt):
        super(SUMEF, self).__init__()
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

        self.attn0 = AttentionModule(nFeat * 2, nFeat)
        self.attn1 = AttentionModule(nFeat * 4, nFeat * 2)
        self.attn2 = AttentionModule(nFeat * 8, nFeat * 4)
        self.attn3 = AttentionModule(nFeat * 16, nFeat * 8)

        # merge
        self.res_module = Merger()

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

        E1 = self.activation(self.conv1(x1))
        E1_0 = self.activation(self.conv1_E0_1(E1))
        E1_0 = self.conv1_E0_2(E1_0)
        E2 = self.activation(self.conv2(x2))
        E2_0 = self.activation(self.conv2_E0_1(E2))
        E2_0 = self.conv2_E0_2(E2_0)
        E1_0 = self.attn0(E1_0, E2_0)

        E1_1 = self.activation(self.conv1_E1_1(E1_0))
        E1_1 = self.conv1_E1_2(E1_1)
        E2_1 = self.activation(self.conv2_E1_1(E2_0))
        E2_1 = self.conv2_E1_2(E2_1)
        E1_1 = self.attn1(E1_1, E2_1)

        E1_2 = self.activation(self.conv1_E2_1(E1_1))
        E1_2 = self.conv1_E2_2(E1_2)
        E2_2 = self.activation(self.conv2_E2_1(E2_1))
        E2_2 = self.conv2_E2_2(E2_2)
        E1_2 = self.attn2(E1_2, E2_2)

        E1_3 = self.activation(self.conv1_E3_1(E1_2))
        E1_3 = self.conv1_E3_2(E1_3)
        E2_3 = self.activation(self.conv2_E3_1(E2_2))
        E2_3 = self.conv2_E3_2(E2_3)
        E1_3 = self.attn3(E1_3, E2_3)

        res_tensor = self.res_module(E1_3, E2_3)
        # print(f"res_tensor: {res_tensor.shape}")

        # Decoder
        D3 = self.activation(self.conv_D3_1(res_tensor))
        D3 = self.conv_D3_2(D3)
        # print(f"D3: {D3.shape}")
        D3_cropped = crop_to_match(D3, E1_2)
        D2 = self.activation(self.conv_D2_1(torch.cat([D3_cropped, E1_2, E2_2], dim=1)))
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

# merging module
class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, dilation, kernel_size=3):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=dilation*(kernel_size-1)//2, bias=True, dilation=dilation)
  def forward(self, x):
    out = F.gelu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        # nDenselayer is the convolution layer number in each DRDB
        # growthRate is the fliter number of each convolution layer
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        dilation = [1, 2, 5]
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate, dilation[i % 3]))
            nChannels_ += growthRate

        # * unpacks the list of modules into the nn.Sequential
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class Merger(nn.Module):
    def __init__(self, nFeat=256, nChannel=3, nDenselayer=6, growthRate=32):
        super(Merger, self).__init__()

        # merging
        self.conv2 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=True)

        # DRDBs 3
        self.RDB1 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = DRDB(nFeat, nDenselayer, growthRate)

        self.RDB3 = DRDB(nFeat, nDenselayer, growthRate)
        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # conv
        self.conv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        F_ = torch.cat((x1, x2), 1) # Z_s

        # there should be LeakyReLU after convolution according to the paper, but it is not implemented here
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        F_4 = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(F_4) # this isn't in the paper
        FGF = self.GFF_3x3(FdLF)
        F_6 = FGF + x2
        # return F_6
        F_7 = self.conv_up(F_6)

        output = self.conv3(F_7)
        # output = torch.stack(torch.split(output, 3 * 4, 1),2)
        # output = nn.functional.sigmoid(output) # this isn't in the paper

        return output


