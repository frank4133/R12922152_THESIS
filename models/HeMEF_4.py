import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def crop_to_match(tensor, target_tensor):
    """
    Crop the input tensor to match the size of the target tensor.
    """
    _, _, h, w = target_tensor.size()
    return tensor[:, :, :h, :w]

class Res_block(nn.Module):
    """改進的 Residual Block with BN"""
    def __init__(self, nFeat, kernel_size=3):
        super(Res_block, self).__init__()
        # Conv -> BN -> Activation 的標準順序
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(nFeat)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nFeat)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual  # 在加法後不需要激活
        return out

class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, ref_img):
        x = torch.cat([img, ref_img], dim=1)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sigmoid(out)
        return out * img

class EvidenceHead(nn.Module):
    """改進的 Evidence Head with BN"""
    def __init__(self, in_ch):
        super().__init__()
        self.econv = nn.Conv2d(in_ch, 2, kernel_size=1, bias=True)

    def forward(self, x):
        e = F.softplus(self.econv(x))  # 使用 softplus 而不是 relu
        alpha = e + 1
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

class HeMEF_4(nn.Module):
    def __init__(self, opt):
        super(HeMEF_4, self).__init__()
        filters_in = 3
        filters_out = 3
        nFeat = 32

        # ===== Encoder1 with BN =====
        self.conv1 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(nFeat)

        self.conv1_E0_1 = nn.Conv2d(nFeat, nFeat, 3, 2, 1, bias=False)
        self.bn1_E0_1 = nn.BatchNorm2d(nFeat)
        self.conv1_E0_2 = Res_block(nFeat)

        self.conv1_E1_1 = nn.Conv2d(nFeat, nFeat * 2, 3, 2, 1, bias=False)
        self.bn1_E1_1 = nn.BatchNorm2d(nFeat * 2)
        self.conv1_E1_2 = Res_block(nFeat * 2)

        self.conv1_E2_1 = nn.Conv2d(nFeat * 2, nFeat * 4, 3, 2, 1, bias=False)
        self.bn1_E2_1 = nn.BatchNorm2d(nFeat * 4)
        self.conv1_E2_2 = Res_block(nFeat * 4)

        self.conv1_E3_1 = nn.Conv2d(nFeat * 4, nFeat * 8, 3, 2, 1, bias=False)
        self.bn1_E3_1 = nn.BatchNorm2d(nFeat * 8)
        self.conv1_E3_2 = Res_block(nFeat * 8)

        # ===== Encoder2 with BN =====
        self.conv2 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nFeat)

        self.conv2_E0_1 = nn.Conv2d(nFeat, nFeat, 3, 2, 1, bias=False)
        self.bn2_E0_1 = nn.BatchNorm2d(nFeat)
        self.conv2_E0_2 = Res_block(nFeat)

        self.conv2_E1_1 = nn.Conv2d(nFeat, nFeat * 2, 3, 2, 1, bias=False)
        self.bn2_E1_1 = nn.BatchNorm2d(nFeat * 2)
        self.conv2_E1_2 = Res_block(nFeat * 2)

        self.conv2_E2_1 = nn.Conv2d(nFeat * 2, nFeat * 4, 3, 2, 1, bias=False)
        self.bn2_E2_1 = nn.BatchNorm2d(nFeat * 4)
        self.conv2_E2_2 = Res_block(nFeat * 4)

        self.conv2_E3_1 = nn.Conv2d(nFeat * 4, nFeat * 8, 3, 2, 1, bias=False)
        self.bn2_E3_1 = nn.BatchNorm2d(nFeat * 8)
        self.conv2_E3_2 = Res_block(nFeat * 8)

        # merge
        self.res_module = Merger()
        self.bn_after_merger = nn.BatchNorm2d(nFeat * 8)  # 在 merger 後加 BN

        # ===== Decoder with BN =====
        # D3 (包含 upsample 的 Sequential，與原始設計一致)
        self.conv_D3_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(nFeat * 8, nFeat * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(nFeat * 4)
        )
        self.conv_D3_2 = Res_block(nFeat * 4)

        # D2 layer
        self.upsample_D2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.cca_D2_E1 = CCA(F_g=nFeat * 4, F_x=nFeat * 4)
        self.cca_D2_E2 = CCA(F_g=nFeat * 4, F_x=nFeat * 4)
        self.conv_D2_1 = nn.Conv2d(nFeat * 12, nFeat * 2, 3, 1, 1, bias=False)
        self.bn_D2_1 = nn.BatchNorm2d(nFeat * 2)
        self.conv_D2_2 = Res_block(nFeat * 2)

        # D1 layer
        self.upsample_D1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.cca_D1_E1 = CCA(F_g=nFeat * 2, F_x=nFeat * 2)
        self.cca_D1_E2 = CCA(F_g=nFeat * 2, F_x=nFeat * 2)
        self.conv_D1_1 = nn.Conv2d(nFeat * 6, nFeat, 3, 1, 1, bias=False)
        self.bn_D1_1 = nn.BatchNorm2d(nFeat)
        self.conv_D1_2 = Res_block(nFeat)

        # D0 layer
        self.upsample_D0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.cca_D0_E1 = CCA(F_g=nFeat, F_x=nFeat)
        self.cca_D0_E2 = CCA(F_g=nFeat, F_x=nFeat)
        self.conv_D0_1 = nn.Conv2d(nFeat * 3, nFeat, 3, 1, 1, bias=False)
        self.bn_D0_1 = nn.BatchNorm2d(nFeat)
        self.conv_D0_2 = Res_block(nFeat)

        # Final D layer
        self.cca_D_E1 = CCA(F_g=nFeat, F_x=nFeat)
        self.cca_D_E2 = CCA(F_g=nFeat, F_x=nFeat)
        self.conv_D_1 = nn.Conv2d(nFeat * 3, nFeat * 3, 3, 1, 1, bias=False)
        self.bn_D_1 = nn.BatchNorm2d(nFeat * 3)
        self.conv_D_2 = Res_block(nFeat * 3)

        # Evidence head
        self.e_head = EvidenceHead(in_ch=nFeat*3)

        # 最終輸出層
        self.conv_out = nn.Conv2d(nFeat * 3, 3, 3, 1, 1, bias=True)
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.opt = opt

    def forward(self, x1, x2):
        # ===== Encoder1 forward =====
        E1 = self.conv1(x1)
        E1 = self.bn1(E1)
        E1 = self.activation(E1)

        E1_0 = self.conv1_E0_1(E1)
        E1_0 = self.bn1_E0_1(E1_0)
        E1_0 = self.activation(E1_0)
        E1_0 = self.conv1_E0_2(E1_0)

        # ===== Encoder2 forward =====
        E2 = self.conv2(x2)
        E2 = self.bn2(E2)
        E2 = self.activation(E2)

        E2_0 = self.conv2_E0_1(E2)
        E2_0 = self.bn2_E0_1(E2_0)
        E2_0 = self.activation(E2_0)
        E2_0 = self.conv2_E0_2(E2_0)

        # E1_1
        E1_1 = self.conv1_E1_1(E1_0)
        E1_1 = self.bn1_E1_1(E1_1)
        E1_1 = self.activation(E1_1)
        E1_1 = self.conv1_E1_2(E1_1)

        # E2_1
        E2_1 = self.conv2_E1_1(E2_0)
        E2_1 = self.bn2_E1_1(E2_1)
        E2_1 = self.activation(E2_1)
        E2_1 = self.conv2_E1_2(E2_1)

        # E1_2
        E1_2 = self.conv1_E2_1(E1_1)
        E1_2 = self.bn1_E2_1(E1_2)
        E1_2 = self.activation(E1_2)
        E1_2 = self.conv1_E2_2(E1_2)

        # E2_2
        E2_2 = self.conv2_E2_1(E2_1)
        E2_2 = self.bn2_E2_1(E2_2)
        E2_2 = self.activation(E2_2)
        E2_2 = self.conv2_E2_2(E2_2)

        # E1_3
        E1_3 = self.conv1_E3_1(E1_2)
        E1_3 = self.bn1_E3_1(E1_3)
        E1_3 = self.activation(E1_3)
        E1_3 = self.conv1_E3_2(E1_3)

        # E2_3
        E2_3 = self.conv2_E3_1(E2_2)
        E2_3 = self.bn2_E3_1(E2_3)
        E2_3 = self.activation(E2_3)
        E2_3 = self.conv2_E3_2(E2_3)

        # Merge
        res_tensor = self.res_module(E1_3, E2_3)
        res_tensor = self.bn_after_merger(res_tensor)

        # ===== Decoder forward =====
        # D3 (conv_D3_1 包含了 upsample)
        D3 = self.conv_D3_1(res_tensor)  # Sequential: upsample -> conv -> bn
        D3 = self.activation(D3)
        D3 = self.conv_D3_2(D3)

        # D2: 使用 D3 做 CCA，然後 concat
        E1_2_refined = self.cca_D2_E1(g=D3, x=E1_2)
        E2_2_refined = self.cca_D2_E2(g=D3, x=E2_2)
        D2 = torch.cat([D3, E1_2_refined, E2_2_refined], dim=1)
        D2 = self.conv_D2_1(D2)
        D2 = self.bn_D2_1(D2)
        D2 = self.activation(D2)
        D2 = self.conv_D2_2(D2)

        # D1: 先 upsample D2，使用未 upsample 的 D2 做 CCA
        D2_up = self.upsample_D2(D2)
        E1_1_refined = self.cca_D1_E1(g=D2, x=E1_1)  # 注意：用 D2，不是 D2_up
        E2_1_refined = self.cca_D1_E2(g=D2, x=E2_1)
        D1 = torch.cat([D2_up, E1_1_refined, E2_1_refined], dim=1)
        D1 = self.conv_D1_1(D1)
        D1 = self.bn_D1_1(D1)
        D1 = self.activation(D1)
        D1 = self.conv_D1_2(D1)

        # D0: 先 upsample D1，使用未 upsample 的 D1 做 CCA
        D1_up = self.upsample_D1(D1)
        E1_0_refined = self.cca_D0_E1(g=D1, x=E1_0)  # 用 D1，不是 D1_up
        E2_0_refined = self.cca_D0_E2(g=D1, x=E2_0)
        D0 = torch.cat([D1_up, E1_0_refined, E2_0_refined], dim=1)
        D0 = self.conv_D0_1(D0)
        D0 = self.bn_D0_1(D0)
        D0 = self.activation(D0)
        D0 = self.conv_D0_2(D0)

        # Final D: 先 upsample D0，使用未 upsample 的 D0 做 CCA
        D0_up = self.upsample_D0(D0)
        E1_refined = self.cca_D_E1(g=D0, x=E1)  # 用 D0，不是 D0_up
        E2_refined = self.cca_D_E2(g=D0, x=E2)
        D = torch.cat([D0_up, E1_refined, E2_refined], dim=1)
        D = self.conv_D_1(D)
        D = self.bn_D_1(D)
        D = self.activation(D)
        D = self.conv_D_2(D)

        # Evidence head
        alpha = self.e_head(D)

        # 權重計算
        if self.opt.evidence_normalization == 'linear':
            total = alpha.sum(dim=1, keepdim=True) + 1e-8
            weights = alpha / total

        elif self.opt.evidence_normalization == 'sigmoid':
            total = alpha.sum(dim=1, keepdim=True) + 1e-8
            weights = torch.sigmoid(alpha / total)

        weight_under = weights[:, 0:1, :, :]
        weight_over = weights[:, 1:2, :, :]
        fusion = weight_under * x1 + weight_over * x2

        return fusion, alpha


# ===== Merger module with BN =====
class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, dilation, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size,
                             padding=dilation*(kernel_size-1)//2, bias=False, dilation=dilation)
        self.bn = nn.BatchNorm2d(growthRate)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = F.gelu(out)
        out = torch.cat((x, out), 1)
        return out

class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        dilation = [1, 2, 5]
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate, dilation[i % 3]))
            nChannels_ += growthRate

        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
        self.bn_1x1 = nn.BatchNorm2d(nChannels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = self.bn_1x1(out)
        out = out + x  # residual connection
        return out

class Merger(nn.Module):
    def __init__(self, nFeat=256, nChannel=3, nDenselayer=6, growthRate=32):
        super(Merger, self).__init__()

        # merging
        self.conv2 = nn.Conv2d(nFeat * 2, nFeat, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(nFeat)

        # DRDBs
        self.RDB1 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = DRDB(nFeat, nDenselayer, growthRate)

        # feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=False)
        self.bn_GFF_1x1 = nn.BatchNorm2d(nFeat)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False)
        self.bn_GFF_3x3 = nn.BatchNorm2d(nFeat)

        # fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False)
        self.bn_up = nn.BatchNorm2d(nFeat)

        # conv
        self.conv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(nFeat)

    def forward(self, x1, x2):
        F_ = torch.cat((x1, x2), 1)

        F_0 = self.conv2(F_)
        F_0 = self.bn2(F_0)
        F_0 = F.gelu(F_0)  # 加入激活函數

        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)

        F_4 = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(F_4)
        FdLF = self.bn_GFF_1x1(FdLF)
        FdLF = F.gelu(FdLF)

        FGF = self.GFF_3x3(FdLF)
        FGF = self.bn_GFF_3x3(FGF)

        F_6 = FGF + x2  # skip connection

        F_7 = self.conv_up(F_6)
        F_7 = self.bn_up(F_7)
        F_7 = F.gelu(F_7)

        output = self.conv3(F_7)
        output = self.bn3(output)

        return output