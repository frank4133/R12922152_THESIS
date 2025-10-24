# -*- coding: utf-8 -*-
"""MEF‑SSIM (multi‑exposure fusion SSIM) — keeps original math, adds minimal
numeric‑stability tweaks (EPS) and auto‑scaling. Now ensures **dtype matches
input** to avoid Float/Double mismatch on conv2d.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

EPS = 1e-8  # tiny constant to prevent 0/0

# -----------------------------------
# Gaussian window helpers
# -----------------------------------

def gaussian(window_size: int, sigma: float):
    g = torch.tensor([
        exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        for x in range(window_size)
    ], dtype=torch.float32)
    return g / g.sum()


def create_window(window_size: int, channel: int, device=None, dtype=torch.float32):
    """Return depth‑wise Gaussian window of shape (channel,1,H,W) in `dtype`."""
    _1d = gaussian(window_size, 1.5).unsqueeze(1)           # (H,1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    win = _2d.expand(channel, 1, window_size, window_size).contiguous()
    return Variable(win.to(device=device, dtype=dtype))

# -----------------------------------
# core MEF‑SSIM
# -----------------------------------

def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, full=False):
    # auto‑scale to 0‑255 if inputs in 0‑1
    if X.max() <= 1.0 and Ys.max() <= 1.0:
        X = X * 255.0
        Ys = Ys * 255.0

    K, C, H, W = Ys.shape

    # --- stats of reference exposures ---
    muY_seq = F.conv2d(Ys, window, padding=ws // 2, groups=C).view(K, C, H, W)
    sigmaY_sq_seq = (F.conv2d(Ys * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) - muY_seq.pow(2))
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # --- stats of test image X ---
    muX = F.conv2d(X, window, padding=ws // 2, groups=C).view(C, H, W)
    sigmaX_sq = (F.conv2d(X * X, window, padding=ws // 2, groups=C).view(C, H, W) - muX.pow(2))

    # --- cross covariance ---
    sigmaXY_seq = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2, groups=C).view(K, C, H, W) - muX.expand_as(muY_seq) * muY_seq

    cs_seq = (2 * sigmaXY_seq + C2) / (sigmaX_sq + sigmaY_sq_seq + C2 + EPS)
    cs_map = torch.gather(cs_seq.view(K, -1), 0, patch_index.view(1, -1)).view(C, H, W)

    # luminance term
    if is_lum:
        lY = muY_seq.mean(dim=(2, 3))
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(-((lY - 0.5) ** 2) / denom_g)[:, None, None, None]
        LY = lG * lL
        muY = (LY * muY_seq).sum(dim=0) / LY.sum(dim=0)
        l_map = (2 * muX * muY + C1) / (muX.pow(2) + muY.pow(2) + C1 + EPS)
    else:
        l_map = torch.ones_like(cs_map)

    if full:
        return l_map.mean(), cs_map.mean()

    qmap = l_map * cs_map
    return qmap.mean()

# -----------------------------------
# wrapper class
# -----------------------------------
class MEFSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super().__init__()
        self.ws = window_size
        self.denom_g = 2 * sigma_g ** 2
        self.denom_l = 2 * sigma_l ** 2
        self.C1 = c1 ** 2
        self.C2 = c2 ** 2
        self.is_lum = is_lum
        self.register_buffer("window", create_window(window_size, channel))

    def forward(self, X, Ys):
        _, C, _, _ = Ys.shape
        # ensure window dtype/device matches input
        if self.window.shape[0] != C or self.window.dtype != Ys.dtype or self.window.device != Ys.device:
            self.window = create_window(self.ws, C, device=Ys.device, dtype=Ys.dtype)
        return _mef_ssim(X, Ys, self.window, self.ws, self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)


def mef_ssim(X, Ys, window_size=11, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
    metric = MEFSSIM(window_size, Ys.shape[1], sigma_g, sigma_l, c1, c2, is_lum).to(X.device, X.dtype)
    return metric(X, Ys)
