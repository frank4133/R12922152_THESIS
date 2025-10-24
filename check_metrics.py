import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import glob
import time
import util.util as util

# 假設你的 MetricsLogger 已經導入
# from your_module import MetricsLogger

def resize_to_match(tensor, target_tensor):
    """
    將 tensor 調整為與 target_tensor 相同的大小
    Args:
        tensor: 要調整大小的張量
        target_tensor: 目標大小的張量
    Returns:
        調整後的張量
    """
    if tensor.shape[-2:] != target_tensor.shape[-2:]:  # 比較 H, W
        target_size = target_tensor.shape[-2:]  # (H, W)
        print(f"Resizing from {tensor.shape[-2:]} to {target_size}")

        # 如果張量是 3D (C, H, W)，先添加 batch 維度
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
            resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
            return resized.squeeze(0)
        # 如果張量是 4D (B, C, H, W)
        elif tensor.dim() == 4:
            return F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

    return tensor

def load_images_from_folder(folder_path):
    """從文件夾加載圖片"""
    # extensions = ['*.jpg', '*pred.jpeg', '*pred.png', '*pred.bmp']
    extensions = ['*.jpg']

    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    image_files.sort()  # 確保順序一致

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    images = []
    filenames = []

    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        images.append(img_tensor)
        filenames.append(os.path.basename(img_path))

    return images, filenames

# 設置四個文件夾路徑
pred_folder = r"/tmp2/frank4133/new_mef/output/HeMEF/0705-123605"        # 預測結果文件夾
label_folder = r"/tmp2/frank4133/datasets/SICE/Test/label"      # 標籤文件夾
input1_folder = r"/tmp2/frank4133/datasets/SICE/Test/under"   # 輸入文件夾1
input2_folder = r"/tmp2/frank4133/datasets/SICE/Test/over"   # 輸入文件夾2

# 設置日誌保存路徑
log_dir = "./outputlogs/hemef/0705-123605"

# 加載圖片
print("Loading images from folders...")
pred_images, pred_names = load_images_from_folder(pred_folder)
label_images, label_names = load_images_from_folder(label_folder)
input1_images, input1_names = load_images_from_folder(input1_folder)
input2_images, input2_names = load_images_from_folder(input2_folder)

print(f"Loaded: {len(pred_images)} pred, {len(label_images)} label, {len(input1_images)} input1, {len(input2_images)} input2")

# 確保數量一致
min_count = min(len(pred_images), len(label_images), len(input1_images), len(input2_images))
print(f"Processing {min_count} images...")

# 初始化你的 MetricsLogger
metrics_logger = util.MetricsLogger()

start_time = time.time()

# 處理每張圖片
for i in range(min_count):
    pred = pred_images[i]
    label = label_images[i]
    input1 = input1_images[i]
    input2 = input2_images[i]

    print(f"\nProcessing image {i+1}/{min_count}: {pred_names[i]}")
    print(f"Original sizes - Pred: {pred.shape}, Label: {label.shape}, Input1: {input1.shape}, Input2: {input2.shape}")

    # 以 pred 為基準，調整其他圖片的大小
    label = resize_to_match(label, pred)
    input1 = resize_to_match(input1, pred)
    input2 = resize_to_match(input2, pred)

    print(f"After resize - Pred: {pred.shape}, Label: {label.shape}, Input1: {input1.shape}, Input2: {input2.shape}")

    # 將兩個input文件夾的圖片堆疊 (在batch維度)
    input1_batch = input1.unsqueeze(0)  # 加上batch維度: (1, 3, H, W)
    input2_batch = input2.unsqueeze(0)  # 加上batch維度: (1, 3, H, W)
    inputs = torch.cat([input1_batch, input2_batch], dim=0)  # 堆疊成 (2, 3, H, W)

    print(f"Final inputs shape: {inputs.shape}")

    # 檢查所有張量的最終大小
    assert pred.shape[-2:] == label.shape[-2:] == input1.shape[-2:] == input2.shape[-2:], \
        f"Size mismatch after resize: pred{pred.shape}, label{label.shape}, input1{input1.shape}, input2{input2.shape}"

    try:
        # 使用你的 MetricsLogger
        metrics_logger.update(pred, label, inputs, log_dir=log_dir, image_name=pred_names[i])
        print(f"✓ Successfully processed {pred_names[i]}")
    except Exception as e:
        print(f"✗ Error processing {pred_names[i]}: {e}")
        continue

total_time = (time.time() - start_time) * 1000

# 顯示最終結果
avg_metrics = metrics_logger.get_average_metrics()
if avg_metrics:
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Total Images Processed: {metrics_logger.count}")
    print(f"Average Processing Time: {total_time / metrics_logger.count:.2f} ms per image")
    print(f"PSNR:     {avg_metrics['psnr']:.4f}")
    print(f"SSIM:     {avg_metrics['ssim']:.4f}")
    print(f"MS-SSIM:  {avg_metrics['msssim']:.4f}")
    print(f"VIF:      {avg_metrics['vif']:.4f}")
    print(f"MEF-SSIM: {avg_metrics['mefssim']:.4f}")
    print("="*50)
else:
    print("No metrics computed!")

# 保存總結
try:
    metrics_logger.save_summary(log_dir, total_time)
    print(f"Results saved to: {log_dir}")
except Exception as e:
    print(f"Error saving results: {e}")

print(f"Total execution time: {total_time/1000:.2f} seconds")