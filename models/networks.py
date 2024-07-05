import torch
import os
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch
from . import slice
from .CTrans import ChannelTransformer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out) + x
        return out

# class Res_block(nn.Module):
#     def __init__(self, nFeat, kernel_size=3):
#         super(Res_block, self).__init__()
#         self.conv1 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)
#         self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=kernel_size, padding=1, bias=True)

#     def forward(self, x):
#         out = F.relu(self.conv1(x), inplace=True)
#         out = self.conv2(out) + x
#         return out

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride = 2, padding=1)
        self.res_block1 = Res_block(out_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.res_block1(x)
        return x

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Linear(F_x, F_x)
        self.mlp_g = nn.Linear(F_g, F_g)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        # g from the lower level of decoder, x from the encoder
        avg_pool_x = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        flat_avg_pool_x = avg_pool_x.view(avg_pool_x.size(0), -1)
        channel_att_x = self.mlp_x(flat_avg_pool_x)
        avg_pool_g = F.avg_pool2d( g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        flat_avg_pool_g = avg_pool_g.view(avg_pool_g.size(0), -1)
        channel_att_g = self.mlp_g(flat_avg_pool_g)
        # print(f"channel_att_x: {channel_att_x.shape}")

        # 相較於原始論文是應用於segmentation，所以輸入只有一個branch
        # 這邊是MEF所以有under, medium, over 三個branch
        # 所以將channel_att_g重複三次
        # 但是也有另外一種方式是將self.mlp_x的輸出變成跟self.mlp_g一樣的維度
        # 這樣就不用重複channel_att_g，不確定何者效果較好，這邊先嘗試重複channel_att_g
        channel_att_g_repeated = channel_att_g.repeat(1, 3)
        # print(f"channel_att_g_repeated: {channel_att_g_repeated.shape}")
        channel_att_sum = (channel_att_x + channel_att_g_repeated)/2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        # 還有一個重大的變化是
        # 原本的decoder是將前一層的decoder跟encoder的skip connection做concat然後upsampling
        # UCT是只將deocder前一層的輸出做upsampling才跟encoder的skip connection去做處理
        self.conv_t = nn.ConvTranspose2d(in_channels, mid_channels, 4, 2, 1, bias=True)
        self.coatt = CCA(F_g=mid_channels, F_x=mid_channels * 3) # because there are 4 branches of inputs
        self.conv_res = Res_block(out_channels, out_channels)

        # added for turning the num of channels of the input of conv_res is the same as the out_channels
        self.conv = nn.Conv2d(mid_channels * 4, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_x):
        # print("input x for conv_t: ", x.shape)
        up = self.conv_t(x)
        skip_x_att = self.coatt(up, skip_x)
        x_concat = torch.cat([up, skip_x_att], dim=1)  # dim 1 is the channel dimension
        x_conv = self.conv(x_concat)
        x_res = self.conv_res(x_conv)
        return x_res


class MEF_dynamic(nn.Module):
    def __init__(self, opt):
        super(MEF_dynamic, self).__init__()
        filters_in = 3
        filters_out = 3
        nFeat = 32

        # encoder for under
        self.conv_under = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv_under_E1 = Encoder(nFeat, nFeat)
        self.conv_under_E2 = Encoder(nFeat, nFeat * 2)
        self.conv_under_E3 = Encoder(nFeat * 2, nFeat * 4)
        self.conv_under_E4 = Encoder(nFeat * 4, nFeat * 8)

        # encoder2
        self.conv_medium = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv_medium_E1 = Encoder(nFeat, nFeat)
        self.conv_medium_E2 = Encoder(nFeat, nFeat * 2)
        self.conv_medium_E3 = Encoder(nFeat * 2, nFeat * 4)
        self.conv_medium_E4 = Encoder(nFeat * 4, nFeat * 8)

        # encoder3
        self.conv_over = nn.Conv2d(filters_in, nFeat, 3, 1, 1, bias=True)
        self.conv_over_E1 = Encoder(nFeat, nFeat)
        self.conv_over_E2 = Encoder(nFeat, nFeat * 2)
        self.conv_over_E3 = Encoder(nFeat * 2, nFeat * 4)
        self.conv_over_E4 = Encoder(nFeat * 4, nFeat * 8)

        # merge
        self.merger = AHDR()

        self.cct = ChannelTransformer(opt, vis=False, channel_num=[nFeat * 3, nFeat * 3, nFeat * 3 * 2, nFeat * 3 * 4])

        # decoder
        self.conv_D3 = Decoder(nFeat * 8, nFeat * 4, nFeat * 4)
        self.conv_D2 = Decoder(nFeat * 4, nFeat * 2, nFeat * 2)
        self.conv_D1 = Decoder(nFeat * 2, nFeat, nFeat)
        self.conv_D0 = Decoder(nFeat * 1, nFeat, nFeat * 3)
        self.conv_out = nn.Conv2d(nFeat * 3, nFeat * 3, 3, 1, 1, bias=True)
        # self.conv_D_2 = Res_block(nFeat * 3)

        # self.conv_out = nn.Conv2d(nFeat * 3, nFeat * 3, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, under, medium, over):

         # Encoder1
        E0_under = self.relu(self.conv_under(under))
        # print(f"E1: {E1.shape}")
        E1_under = self.conv_under_E1(E0_under)
        # print(f"E1_0: {E1_0.shape}")
        E2_under = self.conv_under_E2(E1_under)
        # print(f"E1_1: {E1_1.shape}")
        E3_under = self.conv_under_E3(E2_under)
        # print(f"E1_2: {E1_2.shape}")
        E4_under = self.conv_under_E4(E3_under)
        # print(f"E1_3: {E1_3.shape}")

        # Encoder2
        E0_medium = self.relu(self.conv_medium(medium))
        E1_medium = self.conv_medium_E1(E0_medium)
        E2_medium = self.conv_medium_E2(E1_medium)
        E3_medium = self.conv_medium_E3(E2_medium)
        E4_medium = self.conv_medium_E4(E3_medium)

        # Encoder3
        E0_over = self.relu(self.conv_over(over))
        E1_over = self.conv_over_E1(E0_over)
        E2_over = self.conv_over_E2(E1_over)
        E3_over = self.conv_over_E3(E2_over)
        E4_over = self.conv_over_E4(E3_over)

        # print(f"merged_tensor: {merged_tensor.shape}")

        E0_concat = torch.cat([E0_under, E0_medium, E0_over], dim=1)
        # print(f"E0_concat: {E0_concat.shape}")
        E1_concat = torch.cat([E1_under, E1_medium, E1_over], dim=1)
        # print(f"E1_concat: {E1_concat.shape}")
        E2_concat = torch.cat([E2_under, E2_medium, E2_over], dim=1)
        # print(f"E2_concat: {E2_concat.shape}")
        E3_concat = torch.cat([E3_under, E3_medium, E3_over], dim=1)
        # print(f"E3_concat: {E3_concat.shape}")
        O0, O1, O2, O3, attn_weight = self.cct(E0_concat, E1_concat, E2_concat, E3_concat)

        merged_tensor = self.merger(E4_under, E4_medium, E4_over)

        # Decoder
        # print(f"O3: {O3.shape}")
        D3 = self.conv_D3(merged_tensor, O3)
        # print(f"D3: {D3.shape}")
        # print(f"O2: {O2.shape}")
        D2 = self.conv_D2(D3, O2)
        # print(f"D2: {D2.shape}")
        D1 = self.conv_D1(D2, O1)
        # print(f"D1: {D1.shape}")
        D0 = self.conv_D0(D1, O0)
        # print(f"D0: {D0.shape}")
        D = self.conv_out(D0)
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
        self.se = SELayer(nChannels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = self.se(out)
        out = out + x
        return out

class AHDR(nn.Module):
    def __init__(self, nFeat=256, nChannel=3, nDenselayer=6, growthRate=32):
        super(AHDR, self).__init__()
        self.se = SELayer(nFeat * 3)

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
        F_se = self.se(F_)

        # there should be LeakyReLU after convolution according to the paper, but it is not implemented here
        F_0 = self.conv2(F_se)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        F_4 = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(F_4) # this isn't in the paper
        FGF = self.GFF_3x3(FdLF)
        F_6 = FGF + x2
        F_7 = self.conv_up(F_6)

        output = self.conv3(F_7)
        # output = torch.stack(torch.split(output, 3 * 4, 1),2)
        # output = nn.functional.sigmoid(output) # this isn't in the paper

        return output

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
        self.coeffs = MEF_dynamic(opt)
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
    vgg = Vgg16()
    vgg.load_state_dict(torch.load('./checkpoints/vgg16.weight'))
    vgg = torch.nn.DataParallel(vgg, gpu_ids)
    return vgg

