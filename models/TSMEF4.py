import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from .DDMEF import DDMEF
import util.util as util
from PIL import Image
import os


class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size,
                            padding=(kernel_size-1)//2+1, bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

# Two-Stage AHDR model - AHDR 作為第二階段
class TSMEF4(nn.Module):
    def __init__(self, opt, stage1_model_path=r"./checkpoints/DDMEF/07-19-16-03/500/DDMEF.pth"):
        super(TSMEF4, self).__init__()

        # 基本參數
        nChannel = 3
        nDenselayer = 6
        nFeat = 64
        growthRate = 32
        self.opt = opt
        self.stage1_model = None
        self.stage1_model_path = stage1_model_path

        # ===== 第二階段模型（AHDR）=====
        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)

        # Attention modules
        self.att11 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        self.att31 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # DRDBs 3
        self.RDB1 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = DRDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = DRDB(nFeat, nDenselayer, growthRate)

        # Feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # Fusion
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # Final conv
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.activation = nn.GELU()

    def load_stage1_model(self):
        """載入第一階段的預訓練模型（如 HybridMEF）"""
        # 這裡需要根據您的第一階段模型來調整
        # 假設使用 HybridMEF 或其他 MEF 模型
        self.stage1_model = DDMEF(self.opt)
        checkpoint = torch.load(self.stage1_model_path, map_location='cpu')
        self.stage1_model.load_state_dict(checkpoint)
        self.stage1_model.eval()
        self.stage1_model.cuda(self.opt.gpu_ids[0])
        for param in self.stage1_model.parameters():
            param.requires_grad = False

    def forward(self, x1, x3):
        """
        x1: under-exposed image
        x2: normal-exposed image (will be replaced by stage1 output if use_stage1=True)
        x3: over-exposed image
        """
        with torch.no_grad():
            x2 = self.stage1_model.forward(x1, x3)

        x2 = x2.to(dtype=x1.dtype)
        # 第二階段：使用 AHDR 進行 refinement
        # Feature extraction
        F1_ = self.activation(self.conv1(x1))
        F2_ = self.activation(self.conv1(x2))  # 使用第一階段生成的 x2
        F3_ = self.activation(self.conv1(x3))

        # Attention for under-exposed
        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.activation(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = torch.sigmoid(F1_A)
        F1_ = F1_ * F1_A

        # Attention for over-exposed
        F3_i = torch.cat((F3_, F2_), 1)
        F3_A = self.activation(self.att31(F3_i))
        F3_A = self.att32(F3_A)
        F3_A = torch.sigmoid(F3_A)
        F3_ = F3_ * F3_A

        # Feature concatenation
        F_ = torch.cat((F1_, F2_, F3_), 1)
        F_0 = self.conv2(F_)

        # DRDB blocks
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)

        # Feature fusion
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F2_

        # Final processing
        us = self.conv_up(FDF)
        output = self.conv3(us)
        output = torch.sigmoid(output)

        return output
