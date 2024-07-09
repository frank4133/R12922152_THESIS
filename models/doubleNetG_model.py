import numpy
import numpy as np
import torch
from . import CR
from torch import nn
from .base_model import BaseModel
from . import networks
from MEFSSIM.lossfunction import MEFSSIM
import matplotlib.pyplot as plt
import os

class SingleModel(BaseModel):
    def name(self):
        return 'SingleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        nb = opt.batchSize
        size = opt.fineSize # crop size
        self.opt = opt
        # self.input_img = self.Tensor(nb, 3, size, size)
        # self.input_A_gray = self.Tensor(nb, 1, size, size)
        # self.input_C_gray = self.Tensor(nb, 1, size, size)

        if not opt.isTest:
            if opt.vgg > 0:
                self.vgg_loss = networks.PerceptualLoss(opt)
                self.vgg_loss.cuda()
                self.vgg = networks.load_vgg16("./model", self.gpu_ids) # actually vgg16 is loaded from checkpoint, ./model is not used
                self.vgg.eval()
                for param in self.vgg.parameters():
                    param.requires_grad = False

        # cuda is defined in networks.py
        self.MEF = networks.define_network(networks.MEF_dynamic, opt)

        self.device = next(self.MEF.parameters()).device
        # self.netD_A = networks.define_D()

        self.MEF.train()

        self.old_lr = opt.lr


        self.optimizer = torch.optim.Adam(
            self.MEF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        # self.optimizer_D_A = torch.optim.Adam(
        #     self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.total_loss = 0
        self.loss_count = 0

    # def set_input(self, input):
    #     AtoB = self.opt.which_direction == 'AtoB' # base_options
    #     # input_A size is (1, 3, 320, 320)
    #     input_A = input['A']
    #     input_B = input['B']
    #     input_C = input['C']

    #     # medium exposure
    #     input_M = input['M']

    #     # resize_: in-place resize
    #     # self.input_A size has already been 320 x 320, what the fuck is the below codes used for?????
    #     self.input_A.resize_(input_A.size()).copy_(input_A)
    #     self.input_B.resize_(input_B.size()).copy_(input_B)
    #     self.input_C.resize_(input_C.size()).copy_(input_C)

    #     # medium exposure
    #     self.input_M.resize_(input_M.size()).copy_(input_M)

    #     self.image_paths = input['A_paths' if AtoB else 'B_paths'] # seems not used
    def set_input(self, input):
        self.under = input['A'].clone().detach().to(self.device)
        self.over = input['B'].clone().detach().to(self.device)
        self.gt = input['C'].clone().detach().to(self.device)
        self.medium = input['M'].clone().detach().to(self.device)

    def get_image_paths(self):
        return self.image_paths

    # def backward_basic(self):
    #     self.MSE = torch.nn.MSELoss()
    #     self.loss_basic = self.MSE(self.output1, self.gt)
    #     self.loss_basic.backward()

    def backward_G(self):
        self.loss_DH = 0
        self.loss_DH1 = 0
        self.loss_DH2 = 0
        self.loss_patchDH = 0
        self.loss_patchDH1 = 0
        self.loss_patchDH2 = 0

        # contrastive loss
        # contract weight is 0.1
        if(self.opt.contract_weight > 0):
            self.CTLoss = CR.ContrastLoss()
            self.CTLoss.cuda()

            if(self.opt.fullinput == 0):
                if(self.opt.hasglobal):
                    self.loss_DH1 = self.CTLoss(
                        vgg=self.vgg, a=self.output1, p=self.gt, n=self.under, opt=self.opt, mode='single', group_n=0)
                    self.loss_DH2 = self.CTLoss(
                        vgg=self.vgg, a=self.output1, p=self.gt, n=self.over, opt=self.opt, mode='single', group_n=0)

        if self.opt.patchD_3 > 0:
            self.output1_patch = 0
            self.gt_patch = 0
            self.under_patch = 0
            self.over_patch = 0

            w = self.output1.size(3)
            h = self.output1.size(2)

            for i in range(self.opt.fineSize // self.opt.patchSize):
                self.output1_patch = self.output1[:, :, i * self.opt.patchSize:i * self.opt.patchSize + self.opt.patchSize,
                                         i * self.opt.patchSize:i * self.opt.patchSize + self.opt.patchSize]
                self.gt_patch = self.gt[:, :, i * self.opt.patchSize:i * self.opt.patchSize + self.opt.patchSize,
                                         i * self.opt.patchSize:i * self.opt.patchSize + self.opt.patchSize]
                self.under_patch = self.under[:, :, i * self.opt.patchSize:i * self.opt.patchSize + self.opt.patchSize,
                                          i * self.opt.patchSize:i * self.opt.patchSize + self.opt.patchSize]
                self.over_patch = self.over[:, :, i * self.opt.patchSize:i * self.opt.patchSize + self.opt.patchSize,
                                          i * self.opt.patchSize:i * self.opt.patchSize + self.opt.patchSize]
                self.loss_patchDH1 = self.loss_patchDH1 + self.CTLoss(
                    vgg=self.vgg, a=self.output1_patch, p=self.gt_patch, n=self.under_patch, opt=self.opt, mode='single', group_n=0)
                self.loss_patchDH2 = self.loss_patchDH2 + self.CTLoss(
                    vgg=self.vgg, a=self.output1_patch, p=self.gt_patch, n=self.over_patch, opt=self.opt, mode='single', group_n=0)

        # adversarial loss
        # where the fuck is discriminator loss????
        # gan_weight is 0.1
        self.loss_G_A = 0
        if(self.opt.gan_weight > 0):
            pred_fake = self.netD_A.forward(self.output1)
            self.loss_G_A = self.criterionGAN(pred_fake, True)

            loss_G_A = 0

            # self.opt.D_P_times2 = False
            if not self.opt.D_P_times2:
                self.loss_G_A = self.loss_G_A + loss_G_A # loss_G_A is 0, what is the purpose of this line???????

        # perceptual loss
        # vgg_weight is 0.1
        # self.opt.vgg is 1
        self.loss_vgg_b = 0
        if self.opt.vgg_weight > 0:
            if self.opt.vgg > 0:
                self.loss_vgg_b = self.vgg_loss.compute_vgg_loss(self.vgg,
                                                                 self.output1,
                                                                 self.gt) * self.opt.vgg if self.opt.vgg > 0 else 0

        # MSE loss
        # main_loss_weight is 1
        def charbonnierLoss(predict, target):
            return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + 1e-6)) # epsilon=1e-3
        self.main_loss = 0
        if self.opt.main_loss_weight > 0:
            if self.opt.main_loss_type == 'mse':
                main_loss = nn.MSELoss()
                self.main_loss = main_loss(self.output1, self.gt)
            elif self.opt.main_loss_type == 'l1': # use L1 loss
                main_loss = nn.L1Loss()
                self.main_loss = main_loss(self.output1, self.gt)

        # SSIM loss
        # ssim_loss is 0.1
        if self.opt.ssim_loss > 0:
            batchsize, rows, columns, channels = self.under.shape

            # medium exposure
            # first dimension seems related to input number
            # should be revised to 3
            # but MEFSSIM seems to use output image and GT, why input image is used here?????
            # imgset = numpy.ones(
            #     [3, 3, self.input_A.shape[2], self.input_A.shape[3]])
            myMEFSSIM = MEFSSIM()
            self.ssimscore = 0
            fakeC = torch.permute(self.output1, [0, 2, 3, 1])
            real = torch.permute(self.gt, [0, 2, 3, 1])
            for i in range(0, batchsize):
                # imgset[0, :, :, :] = self.input_A[i].cpu().numpy()
                # imgset[1, :, :, :] = self.input_B[i].cpu().numpy()
                # imgset[2, :, :, :] = self.input_M[i].cpu().numpy()
                # imgset_tensor = torch.tensor(imgset)
                # imgset_tensor = torch.permute(imgset_tensor, [0, 2, 3, 1])

                ssimresult = myMEFSSIM.forward(
                    fakeC[i].unsqueeze(0), real[i].unsqueeze(0))
                if(np.isnan(ssimresult.item()) == 0):
                    self.ssimscore = self.ssimscore + ssimresult

        mw = self.opt.main_loss_weight
        vw = self.opt.vgg_weight
        gw = self.opt.gan_weight
        cw = self.opt.contract_weight
        glr = self.opt.global_local_rate


        # default: fullinput = 0, hasglobal = 1, patchD_3 = 0
        # default: SSIM loss = 0.1, glr = 10
        if(self.opt.fullinput == 0):

            # default: not useing local loss?!?!
            if(self.opt.hasglobal == 1 and self.opt.patchD_3 == 0):
                self.loss_G = mw * self.main_loss + gw*self.loss_G_A + vw * \
                    self.loss_vgg_b + cw/2 * self.loss_DH1 + cw/2*self.loss_DH2
            elif(self.opt.hasglobal == 0 and self.opt.patchD_3 > 0):
                self.loss_G = mw * self.main_loss + gw*self.loss_G_A + vw * \
                    self.loss_vgg_b + cw/2 * self.loss_patchDH1 + cw/2*self.loss_patchDH2
            else: # consider both holistic and local
                self.loss_G = mw * self.main_loss + gw*self.loss_G_A + vw*self.loss_vgg_b + cw/2*(glr/(glr+1)) * self.loss_DH1 + cw/2*(
                    glr/(glr+1))*self.loss_DH2 + cw/2*(1/(glr+1))*self.loss_patchDH1+cw/2*(1/(glr+1))*self.loss_patchDH2

        # Default: batchSize = 1
        if(self.opt.ssim_loss):
            self.loss_G = self.loss_G + self.opt.ssim_loss * \
                (self.opt.batchSize-self.ssimscore)
        self.total_loss += self.loss_G.detach().cpu().numpy()
        self.loss_count += 1
        self.loss_G.backward()

    # def backward_D_basic(self, netD, real, fake, use_ragan):

    #     pred_real = netD.forward(real)
    #     # print(f'> pred_real: {pred_real}')
    #     pred_fake = netD.forward(fake.detach())
    #     # print(f'> pred_fake: {pred_fake}')

    #     # to subtract the mean of the fake/real image, it seems that it is from a technique called RAGAN
    #     # The relativistic discriminator: a key element missing from standard GAN
    #     loss_D = (self.criterionGAN(pred_real - torch.mean(pred_fake), True) +
    #               self.criterionGAN(pred_fake - torch.mean(pred_real), False)) / 2

    #     return loss_D

    # def backward_D_A(self):
    #     fake_C = self.output1
    #     self.loss_D_A = self.backward_D_basic(
    #         self.netD_A, self.gt, fake_C, True)
    #     self.loss_D_A.backward()

    def forward(self):
        # self.gt_gray = Variable(self.input_C_gray)
        # print(f'> net device: {next(self.attentionnet.parameters()).device}')
        self.output1 = self.MEF.forward(self.under, self.medium, self.over)

        # input channels seem correct


    def optimize_parameters(self, epoch):

        self.forward()
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()

        # self.optimizer_D_A.zero_grad()
        # self.backward_D_A()
        # self.optimizer_D_A.step()

    def get_current_errors(self, epoch):
        # G_A = self.loss_G_A
        # MSE = self.main_loss
        # if(self.opt.fullinput == 0):
        #     if(self.opt.hasglobal):
        #         DH1 = self.loss_DH1
        #         DH2 = self.loss_DH2
        # vgg = 0
        # if self.opt.vgg > 0:
        #     if self.opt.vgg_weight > 0:
        #         vgg = self.loss_vgg_b.item() / self.opt.vgg if self.opt.vgg > 0 else 0
        #     mefssim = 0
        #     if self.opt.ssim_loss > 0:
        #         mefssim = self.opt.batchSize-self.ssimscore
        #     if (self.opt.hasglobal == 1 and self.opt.patchD_3 == 0):
        #         return OrderedDict([('MSE', MSE), ('G_A', G_A), ("vgg", vgg), ("DH1", DH1), ("DH2", DH2), ('MEFSSIM', mefssim)])
        total_loss = self.total_loss / self.loss_count
        self.total_loss = 0
        self.loss_count = 0
        return total_loss

    def write_loss(self, save_path, epoch, loss):
        loss_path = save_path + '/loss.txt'
        if not os.path.exists(loss_path):
            with open(loss_path, 'w') as f:
                f.write(f'{epoch} epoch: {loss}\n')
        else:
            with open(loss_path, 'a') as f:
                f.write(f'{epoch} epoch: {loss}\n')

    def save(self, label, save_path):
        # label is epoch
        self.save_network(self.MEF, 'MEF', label, self.gpu_ids, save_path)

        # self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)

    def update_learning_rate(self):

        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %.8f -> %.8f' % (self.old_lr, lr))
        self.old_lr = lr
