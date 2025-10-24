import numpy as np
import torch
from torch import nn
from . import BaseMEF
from .Vgg import VGGPerceptualLoss
from MEFSSIM.lossfunction import MEFSSIM
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import shutil
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from util.mef_ssim import mef_ssim

class BaseModel():
    def __init__(self, opt, network):
        self.opt = opt

        self.network = network
        self.network.apply(self.weights_init)
        if opt.network.startswith("TSMEF"):
            self.network.load_stage1_model()
        self.network.cuda(opt.gpu_ids[0])
        self.device = next(self.network.parameters()).device

        if opt.mode == 'Train':
            if opt.vgg_weight > 0:
                self.vgg = VGGPerceptualLoss(opt.gpu_ids)
            self.old_lr = opt.lr
            self.network.train()
            self.optimizer = torch.optim.Adam(
                self.network.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.total_loss = 0
        self.loss_count = 0

    def set_input(self, input):
        if self.opt.mode == 'Test':
            self.under = input['under'].to(self.device).half()
            self.over = input['over'].to(self.device).half()
            self.label = input['label'].to(self.device).half()
            if self.opt.dataset_name == 'sice3':
                self.medium = input['medium'].to(self.device).half()
        else:
            self.under = input['under'].to(self.device)
            self.over = input['over'].to(self.device)
            self.label = input['label'].to(self.device)
            if self.opt.dataset_name == 'sice3':
                self.medium = input['medium'].to(self.device)

    def get_image_paths(self):
        return self.image_paths

    def make_alpha_star_for2(self, under, over, hdr, epsilon=0.05):
        """
        對 3 張曝光逐像素找與 HDR‑GT 差最小者 → one‑hot +1
        """
        # 計算每張曝光與 HDR GT 的絕對誤差（可改成 MSE）
        err_under = (under - hdr).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        err_over  = (over  - hdr).abs().mean(dim=1, keepdim=True)  # [B,1,H,W]

        # 比較：哪一張誤差小 → 選為可信曝光
        under_better = (err_under < err_over).float()   # [B,1,H,W]
        over_better  = 1.0 - under_better               # [B,1,H,W]

        # 拼成 one-hot 樣式：e.g. [B,2,H,W]
        one_hot = torch.cat([under_better, over_better], dim=1)

        # α★ = one-hot + 1 → softplus 對應的 target
        return (1 - epsilon) * one_hot + epsilon * 0.5 + 1.0

    def make_alpha_star_for3(self, under, medium, over, hdr, epsilon=0.05):
        """
        對 3 張曝光圖像逐像素找與 HDR‑GT 誤差最小者 → one‑hot +1
        """
        # 計算絕對誤差
        err_under = (under - hdr).abs().mean(dim=1, keepdim=True)   # [B,1,H,W]
        err_medium = (medium - hdr).abs().mean(dim=1, keepdim=True) # [B,1,H,W]
        err_over = (over - hdr).abs().mean(dim=1, keepdim=True)     # [B,1,H,W]

        # 堆疊所有誤差成 [B,3,H,W]
        errors = torch.cat([err_under, err_medium, err_over], dim=1)

        # 找出每個像素誤差最小的曝光圖像索引
        min_indices = torch.argmin(errors, dim=1, keepdim=True)  # [B,1,H,W]

        # 建立one-hot編碼 [B,3,H,W]
        one_hot = torch.zeros_like(errors)
        one_hot.scatter_(1, min_indices, 1.0)

        # 產生α★ = one-hot + 1，加上soft regularization
        alpha_star = (1 - epsilon) * one_hot + epsilon / 3 + 1.0

        return alpha_star

    def backward_G(self, epoch):
        # perceptual loss
        # vgg_weight is 0.1
        # self.opt.vgg is 1
        vgg_loss = 0
        if self.opt.vgg_weight > 0:
            vgg_loss = self.vgg.forward(self.output1, self.label)

        # MSE loss
        # main_loss_weight is 1
        main_loss = 0
        if self.opt.main_loss_weight > 0:
            if self.opt.main_loss_type == 'mse':
                main_loss_function = nn.MSELoss()
                main_loss = main_loss_function(self.output1, self.label)
            elif self.opt.main_loss_type == 'l1': # use L1 loss
                main_loss_function = nn.L1Loss()
                main_loss = main_loss_function(self.output1, self.label)

        # SSIM loss
        # ssim_loss is 0.1
        ssim_loss = 0
        if self.opt.ssim_loss_weight > 0:
            # ssim_total = 0.0  # ✅ 使用單獨的累積變量
            # valid_count = 0

            # for i in range(self.under.size(0)):
            #     imgset = torch.stack([self.under[i], self.over[i]], dim=0)
            #     fused = self.output1[i].unsqueeze(0)

            #     ssim_score = 1 - mef_ssim(fused, imgset)  # ✅ 使用不同變量

            #     if torch.isfinite(ssim_score):
            #         ssim_total += ssim_score  # ✅ 累積到總和
            #         valid_count += 1

            # ssim_loss = ssim_total / max(valid_count, 1)  # ✅ 正確平均

            # 計算平均
            # ssim_loss = ssim_loss / max(valid_count, 1)
            def _rgb_to_gray(rgb_tensor):
                """RGB to grayscale conversion"""
                if rgb_tensor.shape[1] != 3:
                    return rgb_tensor

                # 使用標準權重: 0.299*R + 0.587*G + 0.114*B
                weights = torch.tensor([0.299, 0.587, 0.114],
                                    device=rgb_tensor.device,
                                    dtype=rgb_tensor.dtype)
                gray = torch.sum(rgb_tensor * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
                return gray

            output_gray = _rgb_to_gray(self.output1)
            label_gray = _rgb_to_gray(self.label)
            ssim_loss = 1 - ssim(output_gray, label_gray)

        evid_loss = 0
        # if self.opt.evid_weight > 0 and epoch > 100:
            # KL loss
            # self.alpha = torch.clamp(self.alpha, min=1e-2, max=1e4)
            # if self.opt.dataset_name == 'sice3':
            #     # For SICE3 dataset, we use the medium exposure to compute alpha_star
            #     alpha_star = self.make_alpha_star_for3(self.under, self.medium, self.over, self.label)
            # else:
            #     alpha_star = self.make_alpha_star_for2(self.under, self.over, self.label)
            # evid_loss = self.dirichlet_kl(self.alpha, alpha_star)
        if self.opt.evid_weight > 0:
            evid_loss = self.clean_evidence_loss(self.alpha, self.output1, self.label)
            # evid_loss = self.unified_evidence_loss()



        mw = self.opt.main_loss_weight
        vw = self.opt.vgg_weight
        sw = self.opt.ssim_loss_weight
        ew = self.opt.evid_weight

        print(f"main_loss: {main_loss}, vgg_loss: {vgg_loss}, ssim_loss: {ssim_loss}, evid_loss: {evid_loss}")
        loss = mw * main_loss + vw * vgg_loss + sw * ssim_loss + ew * evid_loss

        self.total_loss += loss.detach().cpu().numpy()
        self.loss_count += 1
        loss.backward()
    def clean_evidence_loss(self, alpha, output, hdr):
        """簡潔而完整的證據損失"""
        quality = 1 - (output - hdr).abs().mean(dim=1, keepdim=True)
        evidence = alpha.sum(dim=1, keepdim=True)

        # quality_ranks = quality.flatten(1).argsort(1).argsort(1).float()
        # evidence_ranks = evidence.flatten(1).argsort(1).argsort(1).float()
        # ranking_loss = 1 - F.cosine_similarity(quality_ranks, evidence_ranks, 1).mean()
        quality_flat = quality.flatten(1)  # [B, H*W]
        evidence_flat = evidence.flatten(1)  # [B, H*W]

        # 標準化（零均值、單位方差）
        quality_norm = (quality_flat - quality_flat.mean(1, keepdim=True)) / (quality_flat.std(1, keepdim=True) + 1e-6)
        evidence_norm = (evidence_flat - evidence_flat.mean(1, keepdim=True)) / (evidence_flat.std(1, keepdim=True) + 1e-6)

        # Pearson相關係數（-1表示希望正相關）
        correlation = (quality_norm * evidence_norm).mean(1)
        correlation_loss = -correlation.mean()  # 最大化相關性

        diversity_loss = -torch.log(evidence.flatten(1).std(1) + 1e-6).mean()
        reg_loss = torch.log(evidence + 1).mean()

        # return ranking_loss + 0.1 * diversity_loss + 0.01 * reg_loss
        return correlation_loss + 0.1 * diversity_loss + 0.01 * reg_loss

    def unified_evidence_loss(self):
        quality = 1 - (self.output1 - self.label).abs().mean(dim=1, keepdim=True)

        # 直接在 [0, 1] 空間計算
        lum_under = (0.299 * self.under[:, 0] + 0.587 * self.under[:, 1] + 0.114 * self.under[:, 2]).unsqueeze(1)
        lum_over = (0.299 * self.over[:, 0] + 0.587 * self.over[:, 1] + 0.114 * self.over[:, 2]).unsqueeze(1)
        lum_label = (0.299 * self.label[:, 0] + 0.587 * self.label[:, 1] + 0.114 * self.label[:, 2]).unsqueeze(1)

        # 線性距離
        distance_under = (lum_under - lum_label).abs()
        distance_over = (lum_over - lum_label).abs()

        total_distance = distance_under + distance_over + 1e-6
        weight_under = distance_under / total_distance
        weight_over = distance_over / total_distance

        # Evidence 加權
        evidence_under = self.alpha[:, 0:1, :, :]
        evidence_over = self.alpha[:, 1:2, :, :]
        weighted_evidence = weight_under * evidence_under + weight_over * evidence_over

        # 相關性
        quality_flat = quality.flatten(1)
        evidence_flat = weighted_evidence.flatten(1)

        q_norm = (quality_flat - quality_flat.mean(1, keepdim=True)) / (quality_flat.std(1, keepdim=True) + 1e-6)
        e_norm = (evidence_flat - evidence_flat.mean(1, keepdim=True)) / (evidence_flat.std(1, keepdim=True) + 1e-6)

        correlation_loss = -(q_norm * e_norm).mean()

        diversity_loss = -torch.log(evidence_flat.std(1) + 1e-6).mean()
        reg_loss = torch.log(weighted_evidence + 1).mean()

        return correlation_loss + 0.1 * diversity_loss + 0.01 * reg_loss

    def dirichlet_kl(self, alpha, alpha_star):
        """
        KL( Dir(α) || Dir(α★) )
        """
        S  = alpha.sum(1, keepdim=True)
        S_ = alpha_star.sum(1, keepdim=True)

        # ln B(α)
        lnB_alpha = torch.lgamma(alpha).sum(1,keepdim=True) - torch.lgamma(S)
        # ln B(α★)
        lnB_alpha_star = torch.lgamma(alpha_star).sum(1,keepdim=True) - torch.lgamma(S_)

        dig = torch.digamma(alpha)
        digS = torch.digamma(S)

        # KL = ln(B(α★)/B(α)) + Σ(αᵢ-α★ᵢ)(ψ(αᵢ)-ψ(α₀))
        kl_term1 = lnB_alpha_star - lnB_alpha  # ln(B(α★)/B(α))
        kl_term2 = ((alpha - alpha_star) * (dig - digS)).sum(1, keepdim=True)

        return (kl_term1 + kl_term2).mean()

    def forward(self):
        if self.opt.evidence == 1:
            self.output1, self.alpha = self.network.forward(self.under, self.over)
            return self.output1, self.alpha
        else:
            if self.opt.dataset_name == 'sice3':
                self.output1 = self.network.forward(self.under, self.medium, self.over)
            else:
                self.output1 = self.network.forward(self.under, self.over)
            return self.output1

    def optimize_parameters(self, epoch):
        self.forward()
        self.optimizer.zero_grad()
        self.backward_G(epoch)
        self.optimizer.step()

    def get_current_errors(self, epoch):
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

    def save_network(self, epoch, save_path):
        save_folder = os.path.join(save_path, str(epoch))
        os.makedirs(save_folder, exist_ok=True)
        save_filename = self.opt.network + '.pth'
        save_path = os.path.join(save_folder, save_filename)
        torch.save(self.network.cpu().state_dict(), save_path)
        print(f'Saving the model at {save_path} after epoch {epoch}')
        if len(self.opt.gpu_ids) and torch.cuda.is_available():
            self.network.cuda(self.opt.gpu_ids[0])

    def load_network(self, weight_path):
        save_filename = '%s.pth' % self.opt.network
        save_path = os.path.join(weight_path, save_filename)
        print('Loading the model from %s' % save_path)
        self.network.load_state_dict(torch.load(save_path))

    def continue_train(self):
        self.load_network(self.network, self.opt.continue_path)
        self.network.train()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.continue_epoch > self.opt.niter:
            scalar = self.opt.continue_epoch - self.opt.niter
            self.old.lr = self.opt.lr - scalar * (self.opt.lr / self.opt.niter_decay)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %.8f -> %.8f' % (self.old_lr, lr))
        self.old_lr = lr

    def delete_checkpoints(self, path, training_epochs):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isdir(file_path) and int(filename) < (training_epochs - (self.opt.niter + self.opt.niter_decay) / 2):
                shutil.rmtree(file_path)
                print(f"Deleted {file_path}")

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)