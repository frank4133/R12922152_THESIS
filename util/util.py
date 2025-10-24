import numpy as np
import os
import math
import cv2
from matplotlib import pyplot as plt
import torch
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
    multiscale_structural_similarity_index_measure,
    visual_information_fidelity,
)
from util.mef_ssim import mef_ssim


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)


def tensor2im_2(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (2, 1, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.15

def align_images(im1, im2, img):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    #
    # # Draw top matches
    # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    if h is None or h.shape != (3, 3):
        return img, np.eye(3, dtype=np.float32)

    # Use homography
    height, width, channels = im2.shape
    img_warped = cv2.warpPerspective(img, h, (width, height))

    return img_warped, h

def save_alpha_map(alpha_tensor, save_dir, prefix="alpha", debug=False):
    """
    將模型輸出的 alpha tensor 儲存為灰度圖片（單通道）。
    alpha_tensor: torch.Tensor, shape = [B, C, H, W]，通常 C = 2（under / over）
    """
    import os
    import torch
    import numpy as np
    from PIL import Image

    os.makedirs(save_dir, exist_ok=True)
    alpha_tensor = alpha_tensor.detach().cpu()
    B, C, H, W = alpha_tensor.shape

    if debug:
        print(f"Alpha tensor shape: {alpha_tensor.shape}")
        print(f"Alpha tensor min: {alpha_tensor.min().item():.6f}")
        print(f"Alpha tensor max: {alpha_tensor.max().item():.6f}")
        print(f"Alpha tensor mean: {alpha_tensor.mean().item():.6f}")
        print(f"Alpha tensor std: {alpha_tensor.std().item():.6f}")
        print(f"Contains NaN: {torch.isnan(alpha_tensor).any().item()}")
        print(f"Contains Inf: {torch.isinf(alpha_tensor).any().item()}")
        print("-" * 50)

    # ✅ Inter-channel normalization
    sum_alpha = alpha_tensor.sum(dim=1, keepdim=True) + 1e-8
    norm_alpha = alpha_tensor / sum_alpha  # [B, 2, H, W]

    # ✅ Sum confidence map：用來判斷整體信心（未 normalize，直接儲存）
    raw_sum = alpha_tensor.sum(dim=1)  # shape [B, H, W]

    for b in range(B):
        for c, exposure in enumerate(["under", "over"]):
            alpha_np = norm_alpha[b, c].numpy()

            # 將值轉換到 0-255 範圍
            # 注意：norm_alpha 已經在 0-1 範圍內
            alpha_uint8 = (alpha_np * 255).astype(np.uint8)

            # 儲存為灰度圖
            filename = f"{prefix}_b{b}_{exposure}.png"
            filepath = os.path.join(save_dir, filename)
            Image.fromarray(alpha_uint8, mode='L').save(filepath)

            if debug and b == 0:  # 只在第一個 batch 顯示
                print(f"Saved {exposure}: min={alpha_np.min():.3f}, max={alpha_np.max():.3f}")

        # 儲存信心總和圖
        sum_np = raw_sum[b].numpy()
        sum_norm = (sum_np - sum_np.min()) / (sum_np.max() - sum_np.min() + 1e-8)
        sum_uint8 = (sum_norm * 255).astype(np.uint8)

        sum_path = os.path.join(save_dir, f"{prefix}_b{b}_sum.png")
        Image.fromarray(sum_uint8, mode='L').save(sum_path)

        if debug and b == 0:
            print(f"Saved sum: min={sum_norm.min():.3f}, max={sum_norm.max():.3f}")

class MetricsLogger:
    def __init__(self):
        self.count = 0
        self.psnr = self.ssim = self.msssim = self.vif = self.mefssim = 0.0
        self.individual_results = []  # 儲存每張圖片的結果

    # ---------- utils ----------
    def _to_tensor(self, x):
        return x if torch.is_tensor(x) else torch.as_tensor(x)

    @staticmethod
    def _rgb2gray(img):               # img : (B,3,H,W)
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b  # → (B,1,H,W)

    def _write_individual_metrics(self, metrics_dict, log_dir, image_idx):
        """將單張圖片的指標寫入文件"""
        if log_dir is None:
            return

        # 確保目錄存在
        os.makedirs(log_dir, exist_ok=True)

        # 寫入個別指標文件
        metrics_file = os.path.join(log_dir, 'individual_metrics.txt')

        # 如果是第一次寫入，先寫標題
        if image_idx == 0 and (not os.path.exists(metrics_file) or os.path.getsize(metrics_file) == 0):
            with open(metrics_file, 'w') as f:
                f.write("Image_Index\tPSNR\tSSIM\tMS-SSIM\tVIF\tMEF-SSIM\n")

        # 寫入數據
        with open(metrics_file, 'a') as f:
            f.write(f"{image_idx}\t{metrics_dict['psnr']:.4f}\t{metrics_dict['ssim']:.4f}\t"
                   f"{metrics_dict['msssim']:.4f}\t{metrics_dict['vif']:.4f}\t{metrics_dict['mefssim']:.4f}\n")

        # 同時寫入 CSV 格式（方便後續分析）
        csv_file = os.path.join(log_dir, 'individual_metrics.csv')

        # 如果是第一次寫入 CSV，先寫標題
        if image_idx == 0 and (not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0):
            import csv
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image_Index', 'PSNR', 'SSIM', 'MS-SSIM', 'VIF', 'MEF-SSIM'])

        # 寫入 CSV 數據
        import csv
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([metrics_dict['image_name'], f"{metrics_dict['psnr']:.4f}", f"{metrics_dict['ssim']:.4f}",
                           f"{metrics_dict['msssim']:.4f}", f"{metrics_dict['vif']:.4f}",
                           f"{metrics_dict['mefssim']:.4f}"])

    # ---------- main ----------
    def update(self, pred, label, inputs, log_dir=None, image_name=None):
        # 1) tensor & batch-dim
        pred, label = map(self._to_tensor, (pred, label))
        if pred.dim() == 3:  pred = pred.unsqueeze(0)
        if label.dim() == 3: label = label.unsqueeze(0)

        # 2) device 同步
        device = pred.device if pred.device == label.device else (
            pred.device if pred.is_cuda else label.device)
        pred, label = pred.to(device), label.to(device)

        def safe_normalize(tensor, name):
            max_val = float(tensor.max())
            min_val = float(tensor.min())

            # if max_val > 10.0:  # 假設是 [0,255] 範圍
            #     return tensor / 255.0
            if max_val > 1.0 or min_val < 0.0:  # 輕微超出 [0,1]
                return torch.clamp(tensor, 0.0, 1.0)
            else:  # 已經在合理範圍
                return tensor

        pred = safe_normalize(pred, "pred").float()
        label = safe_normalize(label, "label").float()
        inputs = safe_normalize(inputs, "inputs").float()

        # 4) 轉灰階（僅 SSIM / MS-SSIM / VIF 用）
        if pred.shape[1] == 3:
            pred_gray  = self._rgb2gray(pred)
            label_gray = self._rgb2gray(label)
        else:           # 已是 1ch
            pred_gray, label_gray = pred, label

        # 5) 指標計算
        psnr_val   = peak_signal_noise_ratio(pred,  label)
        ssim_val   = structural_similarity_index_measure(
            pred_gray,  label_gray)
        msssim_val = multiscale_structural_similarity_index_measure(
            pred_gray,  label_gray)
        vif_val    = visual_information_fidelity(pred_gray, label_gray,
                                                 sigma_n_sq=2.0)
        mef_val = mef_ssim(pred, inputs)

        # 6) 累積
        self.psnr     += psnr_val.item()
        self.ssim     += ssim_val.item()
        self.msssim   += msssim_val.item()
        self.vif      += vif_val.item()
        self.mefssim  += float(mef_val) if not torch.is_tensor(mef_val) else mef_val.item()

        # 7) 儲存個別結果
        individual_metrics = {
            'psnr': psnr_val.item(),
            'ssim': ssim_val.item(),
            'msssim': msssim_val.item(),
            'vif': vif_val.item(),
            'mefssim': float(mef_val) if not torch.is_tensor(mef_val) else mef_val.item(),
            'image_name': image_name if image_name else f"image_{self.count}"
        }
        self.individual_results.append(individual_metrics)

        # 8) 寫入個別指標
        if log_dir:
            self._write_individual_metrics(individual_metrics, log_dir, self.count)

        # 9) 即時列印
        print(
            f"[{individual_metrics['image_name']}] "
            f"PSNR:{psnr_val:.4f}  SSIM:{ssim_val:.4f}  MS-SSIM:{msssim_val:.4f}  "
            f"VIF:{vif_val:.4f}  MEF-SSIM:{mef_val:.4f}"
        )

        self.count += 1

    def get_average_metrics(self):
        """取得平均指標"""
        if self.count == 0:
            return None

        return {
            'psnr': self.psnr / self.count,
            'ssim': self.ssim / self.count,
            'msssim': self.msssim / self.count,
            'vif': self.vif / self.count,
            'mefssim': self.mefssim / self.count
        }

    def save_summary(self, log_dir, total_time):
        """儲存總結報告"""
        if log_dir is None or self.count == 0:
            return

        os.makedirs(log_dir, exist_ok=True)

        # 計算平均值
        avg_metrics = self.get_average_metrics()

        # 寫入總結文件
        summary_file = os.path.join(log_dir, 'metrics_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("=== Metrics Summary ===\n")
            f.write(f"Total Images: {self.count}\n\n")
            f.write("Average Metrics:\n")
            f.write(f"Average Time: {total_time / self.count:.4f} ms\n")
            f.write(f"PSNR:     {avg_metrics['psnr']:.4f}\n")
            f.write(f"SSIM:     {avg_metrics['ssim']:.4f}\n")
            f.write(f"MS-SSIM:  {avg_metrics['msssim']:.4f}\n")
            f.write(f"VIF:      {avg_metrics['vif']:.4f}\n")
            f.write(f"MEF-SSIM: {avg_metrics['mefssim']:.4f}\n\n")

            # 計算標準差
            if len(self.individual_results) > 1:
                import numpy as np
                psnr_values = [r['psnr'] for r in self.individual_results]
                ssim_values = [r['ssim'] for r in self.individual_results]
                msssim_values = [r['msssim'] for r in self.individual_results]
                vif_values = [r['vif'] for r in self.individual_results]
                mefssim_values = [r['mefssim'] for r in self.individual_results]

                f.write("Standard Deviation:\n")
                f.write(f"PSNR:     {np.std(psnr_values):.4f}\n")
                f.write(f"SSIM:     {np.std(ssim_values):.4f}\n")
                f.write(f"MS-SSIM:  {np.std(msssim_values):.4f}\n")
                f.write(f"VIF:      {np.std(vif_values):.4f}\n")
                f.write(f"MEF-SSIM: {np.std(mefssim_values):.4f}\n\n")

                # 最好和最差的結果
                f.write("Best Results:\n")
                best_psnr_idx = np.argmax(psnr_values)
                best_ssim_idx = np.argmax(ssim_values)
                f.write(f"PSNR:     {self.individual_results[best_psnr_idx]['image_name']} ({psnr_values[best_psnr_idx]:.4f})\n")
                f.write(f"SSIM:     {self.individual_results[best_ssim_idx]['image_name']} ({ssim_values[best_ssim_idx]:.4f})\n\n")

                f.write("Worst Results:\n")
                worst_psnr_idx = np.argmin(psnr_values)
                worst_ssim_idx = np.argmin(ssim_values)
                f.write(f"PSNR:     {self.individual_results[worst_psnr_idx]['image_name']} ({psnr_values[worst_psnr_idx]:.4f})\n")
                f.write(f"SSIM:     {self.individual_results[worst_ssim_idx]['image_name']} ({ssim_values[worst_ssim_idx]:.4f})\n")