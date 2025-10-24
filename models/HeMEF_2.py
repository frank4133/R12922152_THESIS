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
class EvidenceHead(nn.Module):
    """3‑通道 evidence → softplus(+1) 產生 Dirichlet α_low/mid/high"""
    def __init__(self, in_ch):
        super().__init__()
        self.econv = nn.Conv2d(in_ch, 2, kernel_size=1, bias=True)

    def forward(self, x):
        e = F.softplus(self.econv(x))          # evidence ≥ 0
        alpha = e + 1                          # α = e + 1
        return alpha

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block - Channel-wise Cross Attention
    用於處理skip connection中encoder和decoder特徵的不一致性
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: decoder特徵 (來自upsampling後)
        # x: encoder特徵 (來自skip connection)

        # channel-wise attention
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)

        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)

        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class HeMEF_2(nn.Module):
    def __init__(self, opt):
        super(HeMEF_2, self).__init__()
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
        self.res_module = Merger()

        # decoder - 使用 Upsample + Conv (方案2) with CCA
        self.conv_D3_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nFeat * 8, nFeat * 4, 3, 1, 1, bias=True)
        )
        self.conv_D3_2 = Res_block(nFeat * 4)

        # D2 layer: upsample -> CCA -> conv -> res_block
        self.upsample_D2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # CCA modules for D2 skip connections (both use nFeat * 4)
        self.cca_D2_E1 = CCA(F_g=nFeat * 4, F_x=nFeat * 4)  # for E1_2
        self.cca_D2_E2 = CCA(F_g=nFeat * 4, F_x=nFeat * 4)  # for E2_2
        self.conv_D2_1 = nn.Conv2d(nFeat * 12, nFeat * 2, 3, 1, 1, bias=True)
        self.conv_D2_2 = Res_block(nFeat * 2)

        # D1 layer: upsample -> CCA -> conv -> res_block
        self.upsample_D1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # CCA modules for D1 skip connections (both use nFeat * 2)
        self.cca_D1_E1 = CCA(F_g=nFeat * 2, F_x=nFeat * 2)  # for E1_1
        self.cca_D1_E2 = CCA(F_g=nFeat * 2, F_x=nFeat * 2)  # for E2_1
        self.conv_D1_1 = nn.Conv2d(nFeat * 6, nFeat, 3, 1, 1, bias=True)
        self.conv_D1_2 = Res_block(nFeat)

        # D0 layer: upsample -> CCA -> conv -> res_block
        self.upsample_D0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # CCA modules for D0 skip connections (both use nFeat)
        self.cca_D0_E1 = CCA(F_g=nFeat, F_x=nFeat)  # for E1_0
        self.cca_D0_E2 = CCA(F_g=nFeat, F_x=nFeat)  # for E2_0
        self.conv_D0_1 = nn.Conv2d(nFeat * 3, nFeat, 3, 1, 1, bias=True)
        self.conv_D0_2 = Res_block(nFeat)

        # CCA modules for final D layer skip connections
        self.cca_D_E1 = CCA(F_g=nFeat, F_x=nFeat)  # for E1
        self.cca_D_E2 = CCA(F_g=nFeat, F_x=nFeat)  # for E2
        self.conv_D_1 = nn.Conv2d(nFeat * 3, nFeat * 3, 3, 1, 1, bias=True)
        self.conv_D_2 = Res_block(nFeat * 3)

        self.e_head = EvidenceHead(in_ch=nFeat*3)

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
        # E1_0 = self.attn0(E1_0, E2_0)

        E1_1 = self.activation(self.conv1_E1_1(E1_0))
        E1_1 = self.conv1_E1_2(E1_1)
        E2_1 = self.activation(self.conv2_E1_1(E2_0))
        E2_1 = self.conv2_E1_2(E2_1)
        # E1_1 = self.attn1(E1_1, E2_1)

        E1_2 = self.activation(self.conv1_E2_1(E1_1))
        E1_2 = self.conv1_E2_2(E1_2)
        E2_2 = self.activation(self.conv2_E2_1(E2_1))
        E2_2 = self.conv2_E2_2(E2_2)
        # E1_2 = self.attn2(E1_2, E2_2)

        E1_3 = self.activation(self.conv1_E3_1(E1_2))
        E1_3 = self.conv1_E3_2(E1_3)
        E2_3 = self.activation(self.conv2_E3_1(E2_2))
        E2_3 = self.conv2_E3_2(E2_3)
        # E1_3 = self.attn3(E1_3, E2_3)

        res_tensor = self.res_module(E1_3, E2_3)
        # print(f"res_tensor: {res_tensor.shape}")

        D3 = self.activation(self.conv_D3_1(res_tensor))
        D3 = self.conv_D3_2(D3)
        # print(f"D3: {D3.shape}")

        # D2: upsample -> CCA -> concat -> conv -> res_block
        # Apply CCA to encoder features (all features have nFeat * 4 channels)
        E1_2_refined = self.cca_D2_E1(g=D3, x=E1_2)  # nFeat * 4
        E2_2_refined = self.cca_D2_E2(g=D3, x=E2_2)  # nFeat * 4
        # Concatenate and reduce channels
        D2 = torch.cat([D3, E1_2_refined, E2_2_refined], dim=1)  # nFeat * 12
        D2 = self.activation(self.conv_D2_1(D2))  # nFeat * 12 -> nFeat * 2
        D2 = self.conv_D2_2(D2)
        # print(f"D2: {D2.shape}")

        # D1: upsample -> CCA -> concat -> conv -> res_block
        D2 = self.upsample_D2(D2)  # nFeat * 2, spatial size x2
        # Apply CCA to encoder features (all features have nFeat * 2 channels)
        E1_1_refined = self.cca_D1_E1(g=D2, x=E1_1)  # nFeat * 2
        E2_1_refined = self.cca_D1_E2(g=D2, x=E2_1)  # nFeat * 2
        # Concatenate and reduce channels
        D1 = torch.cat([D2, E1_1_refined, E2_1_refined], dim=1)  # nFeat * 6
        D1 = self.activation(self.conv_D1_1(D1))  # nFeat * 6 -> nFeat
        D1 = self.conv_D1_2(D1)
        # print(f"D1: {D1.shape}")

        # D0: upsample -> CCA -> concat -> conv -> res_block
        D1 = self.upsample_D1(D1)  # nFeat, spatial size x2
        # Apply CCA to encoder features (all features have nFeat channels)
        E1_0_refined = self.cca_D0_E1(g=D1, x=E1_0)  # nFeat
        E2_0_refined = self.cca_D0_E2(g=D1, x=E2_0)  # nFeat
        # Concatenate and reduce channels
        D0 = torch.cat([D1, E1_0_refined, E2_0_refined], dim=1)  # nFeat * 3
        D0 = self.activation(self.conv_D0_1(D0))  # nFeat * 3 -> nFeat
        D0 = self.conv_D0_2(D0)
        # print(f"D0: {D0.shape}")

        # Final D layer with CCA
        # Apply CCA to encoder features
        D0 = self.upsample_D0(D0)  # nFeat, spatial size x2
        E1_refined = self.cca_D_E1(g=D0, x=E1)  # nFeat
        E2_refined = self.cca_D_E2(g=D0, x=E2)  # nFeat
        D = self.activation(self.conv_D_1(torch.cat([D0, E1_refined, E2_refined], dim=1)))  # nFeat * 3
        D = self.conv_D_2(D)

        # print(f"output: {output.shape}")
        alpha = self.e_head(D)
        out = self.sigmoid(self.conv_out(D))
        #return D3, D2, D1, D0, D, out # for training
        return out, alpha

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


