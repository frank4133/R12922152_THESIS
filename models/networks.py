import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
from . import slice

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Res_block(nn.Module):
    def __init__(self, nFeat, kernel_size=3):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out) + x
        return out


class MEF_dynamic(nn.Module):
    def __init__(self):
        super(MEF_dynamic, self).__init__()
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

        # encoder3
        self.conv3 = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv3_E0_1 = nn.Conv2d(nFeat, nFeat, 3, 2, 1, bias=True)
        self.conv3_E0_2 = Res_block(nFeat)
        self.conv3_E1_1 = nn.Conv2d(nFeat, nFeat * 2, 3, 2, 1, bias=True)
        self.conv3_E1_2 = Res_block(nFeat * 2)
        self.conv3_E2_1 = nn.Conv2d(nFeat * 2, nFeat * 4, 3, 2, 1, bias=True)
        self.conv3_E2_2 = Res_block(nFeat * 4)
        self.conv3_E3_1 = nn.Conv2d(nFeat * 4, nFeat * 8, 3, 2, 1, bias=True)
        self.conv3_E3_2 = Res_block(nFeat * 8)

        # merge
        self.res_module = AHDR()

        # decoder
        self.conv_D3_1 = nn.ConvTranspose2d(nFeat * 8, nFeat * 4, 4, 2, 1, bias=True)
        self.conv_D3_2 = Res_block(nFeat * 4)
        self.conv_D2_1 = nn.ConvTranspose2d(nFeat * 16, nFeat * 2, 4, 2, 1, bias=True)
        self.conv_D2_2 = Res_block(nFeat * 2)
        self.conv_D1_1 = nn.ConvTranspose2d(nFeat * 8, nFeat, 4, 2, 1, bias=True)
        self.conv_D1_2 = Res_block(nFeat)
        self.conv_D0_1 = nn.ConvTranspose2d(nFeat * 4, nFeat, 4, 2, 1, bias=True)
        self.conv_D0_2 = Res_block(nFeat)
        self.conv_D_1 = nn.Conv2d(nFeat * 4, nFeat * 3, 3, 1, 1, bias=True)
        self.conv_D_2 = Res_block(nFeat * 3)

        self.conv_out = nn.Conv2d(nFeat * 3, nFeat * 3, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3):

         # Encoder1
        E1 = self.relu(self.conv1(x1))
        # print(f"E1: {E1.shape}")
        E1_0 = self.relu(self.conv1_E0_1(E1))
        E1_0 = self.conv1_E0_2(E1_0)
        # print(f"E1_0: {E1_0.shape}")
        E1_1 = self.relu(self.conv1_E1_1(E1_0))
        E1_1 = self.conv1_E1_2(E1_1)
        # print(f"E1_1: {E1_1.shape}")
        E1_2 = self.relu(self.conv1_E2_1(E1_1))
        E1_2 = self.conv1_E2_2(E1_2)
        # print(f"E1_2: {E1_2.shape}")
        E1_3 = self.relu(self.conv1_E3_1(E1_2))
        E1_3 = self.conv1_E3_2(E1_3)
        # print(f"E1_3: {E1_3.shape}")

        # Encoder2
        E2 = self.relu(self.conv2(x2))
        E2_0 = self.relu(self.conv2_E0_1(E2))
        E2_0 = self.conv2_E0_2(E2_0)
        E2_1 = self.relu(self.conv2_E1_1(E2_0))
        E2_1 = self.conv2_E1_2(E2_1)
        E2_2 = self.relu(self.conv2_E2_1(E2_1))
        E2_2 = self.conv2_E2_2(E2_2)
        E2_3 = self.relu(self.conv2_E3_1(E2_2))
        E2_3 = self.conv2_E3_2(E2_3)

        # Encoder3
        E3 = self.relu(self.conv3(x3))
        E3_0 = self.relu(self.conv3_E0_1(E3))
        E3_0 = self.conv3_E0_2(E3_0)
        E3_1 = self.relu(self.conv3_E1_1(E3_0))
        E3_1 = self.conv3_E1_2(E3_1)
        E3_2 = self.relu(self.conv3_E2_1(E3_1))
        E3_2 = self.conv3_E2_2(E3_2)
        E3_3 = self.relu(self.conv3_E3_1(E3_2))
        E3_3 = self.conv3_E3_2(E3_3)

        res_tensor = self.res_module(E1_3, E2_3, E3_3)
        # print(f"res_tensor: {res_tensor.shape}")

        # Decoder
        D3 = self.relu(self.conv_D3_1(res_tensor))
        D3 = self.conv_D3_2(D3)
        # print(f"D3: {D3.shape}")
        D2 = self.relu(self.conv_D2_1(torch.cat([D3, E1_2, E2_2, E3_2], dim=1)))
        D2 = self.conv_D2_2(D2)
        # print(f"D2: {D2.shape}")
        D1 = self.relu(self.conv_D1_1(torch.cat([D2, E1_1, E2_1, E3_1], dim=1)))
        D1 = self.conv_D1_2(D1)
        # print(f"D1: {D1.shape}")
        D0 = self.relu(self.conv_D0_1(torch.cat([D1, E1_0, E2_0, E3_0], dim=1)))
        D0 = self.conv_D0_2(D0)
        # print(f"D0: {D0.shape}")
        D = self.relu(self.conv_D_1(torch.cat([D0, E1, E2, E3], dim=1)))
        D = self.conv_D_2(D)
        # print(f"D: {D.shape}")

        output = torch.stack(torch.split(D, 3 * 4, 1),2)
        # print(f"output: {output.shape}")
        # out = self.sigmoid(self.conv_out(D))
        #return D3, D2, D1, D0, D, out # for training
        return output

# merging module
class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
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
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate))
            nChannels_ += growthRate

        # * unpacks the list of modules into the nn.Sequential
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out

class AHDR(nn.Module):
    def __init__(self, nFeat=256, nChannel=3, nDenselayer=6, growthRate=32):
        super(AHDR, self).__init__()

        # merging
        self.conv2 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=3, padding=1, bias=True)

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

    def forward(self, x1, x2, x3):
        F_ = torch.cat((x1, x2, x3), 1) # Z_s

        # there should be LeakyReLU after convolution according to the paper, but it is not implemented here
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        F_4 = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(F_4) # this isn't in the paper
        FGF = self.GFF_3x3(FdLF)
        F_6 = FGF + x3
        return F_6
        # F_7 = self.conv_up(F_6)

        # output = self.conv3(F_7)
        # output = torch.stack(torch.split(output, 3 * 4, 1),2)
        # output = nn.functional.sigmoid(output) # this isn't in the paper

        # return output

def define_network(network_class, opt):
    use_gpu = len(opt.gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    net = network_class(opt)
    net.apply(weights_init)
    net.cuda(opt.gpu_ids[0])
    return net

class GuideNN(nn.Module):
    def __init__(self, opt):
        super(GuideNN, self).__init__()

        # guide complexity = 16
        self.conv1 = nn.Conv2d(3, opt.guide_complexity, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(opt.guide_complexity, 1, kernel_size=1, padding=0) #nn.Tanh nn.Sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv2(self.conv1(x)))#.squeeze(1)

class Slice(nn.Module):
    def __init__(self):
        super(Slice, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        bilateral_grid = bilateral_grid.permute(0,3,4,2,1)
        guidemap = guidemap.squeeze(1)
        # grid: The bilateral grid with shape (gh, gw, gd, gc).
        # guide: A guide image with shape (h, w). Values must be in the range [0, 1].
        coeefs = slice.bilateral_slice(bilateral_grid, guidemap).permute(0,3,1,2)
        return coeefs

class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):

        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        # out_channels = []
        # for chan in range(n_out):
        #     ret = scale[:, :, :, chan, 0]*input_image[:, :, :, 0]
        #     for chan_i in range(1, n_in):
        #         ret += scale[:, :, :, chan, chan_i]*input_image[:, :, :, chan_i]
        #     if has_affine_term:
        #         ret += offset[:, :, :, chan]
        #     ret = tf.expand_dims(ret, 3)
        #     out_channels.append(ret)

        # ret = tf.concat(out_channels, 3)
        """
            R = r1[0]*r2 + r1[1]*g2 + r1[2]*b3 +r1[3]
        """

        # print(coeff.shape)
        # R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        # G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        # B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]
        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 9:10, :, :]
        G = torch.sum(full_res_input * coeff[:, 3:6, :, :], dim=1, keepdim=True) + coeff[:, 10:11, :, :]
        B = torch.sum(full_res_input * coeff[:, 6:9, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)

class MEFPointwiseNN(nn.Module):

    def __init__(self, opt):
        super(MEFPointwiseNN, self).__init__()
        self.coeffs = MEF_dynamic()
        self.guide = GuideNN(opt=opt)
        self.slice = Slice()
        self.apply_coeffs = ApplyCoeffs()
        # self.bsa = bsa.BilateralSliceApply()

    def forward(self, lowU, lowM, lowO, full):
        coeffs = self.coeffs(lowU, lowM, lowO)
        guide = self.guide(full)
        slice_coeffs = self.slice(coeffs, guide)
        out = self.apply_coeffs(slice_coeffs, full)
        # out = bsa.bsa(coeffs,guide,full)
        return out


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X, opt):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        if opt.vgg_choose != "no_maxpool":
            h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)

        if opt.vgg_choose != "no_maxpool":
            if opt.vgg_maxpooling:
                h = F.max_pool2d(h, kernel_size=2, stride=2)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)
        if opt.vgg_choose == "relu5_1":
            return relu5_1


def vgg_preprocess(batch, opt):
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    return batch


class PerceptualLoss(nn.Module):
    def __init__(self, opt):
        super(PerceptualLoss, self).__init__()
        self.opt = opt
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img, self.opt)
        target_vgg = vgg_preprocess(target, self.opt)
        img_fea = vgg(img_vgg, self.opt)
        target_fea = vgg(target_vgg, self.opt)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)


def load_vgg16(model_dir, gpu_ids):
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    vgg = Vgg16()
    vgg.load_state_dict(torch.load('./checkpoints/vgg16.weight'))
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg

