import os.path
import torchvision.transforms.v2 as transforms
from PIL import Image
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from util.util import align_images
from data.base_dataset import BaseDataset
from skimage import exposure
import math

class SICEDataset(BaseDataset):
    def __init__(self, opt):
        self.opt = opt
        self.root_dir = os.path.join(r'../datasets/SICE', opt.mode + '2')
        self.under_dir = 'under'
        self.over_dir = 'over'
        self.label_dir = 'label'
        self.under_path = []
        self.over_path = []
        self.label_path = []

        self.scenes_list = sorted(os.listdir(os.path.join(self.root_dir, self.label_dir)))
        for scene in self.scenes_list:
            self.over_path.append(os.path.join(self.root_dir, self.over_dir, scene))
            self.label_path.append(os.path.join(self.root_dir, self.label_dir, scene ))
            self.under_path.append(os.path.join(self.root_dir, self.under_dir, scene))

        if (self.opt.mode == 'Train'):
            self.trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),   # 随机水平翻转
                transforms.RandomVerticalFlip(),     # 随机垂直翻转
                transforms.RandomCrop((self.opt.crop_size, self.opt.crop_size)),    # 随机裁剪到指定大小
                transforms.ToTensor()               # 将PIL图像转换为张量
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        img_over = Image.open(self.over_path[index]).convert('RGB')
        img_label = Image.open(self.label_path[index]).convert('RGB')
        img_under = Image.open(self.under_path[index]).convert('RGB')

        # To make sure that the concate image can be divided by 16
        if (self.opt.mode == 'Test'):
            w = img_label.size[0]
            h = img_label.size[1]
            new_w = w - w % 64
            new_h = h - h % 64
            resample_filter = Image.Resampling.LANCZOS
            img_over = np.array(img_over.resize((new_w // 4, new_h // 4), resample=resample_filter))
            img_label = np.array(img_label.resize((new_w // 4, new_h // 4), resample=resample_filter))
            img_under = np.array(img_under.resize((new_w // 4, new_h // 4), resample=resample_filter))


        if (self.opt.warp == 1):
            img_match = np.zeros_like(img_under)
            for i in range(3):
                img_match[..., i] = exposure.match_histograms(img_under[..., i], img_over[..., i])
            img_under, h = align_images(np.array(img_match), np.array(img_over), np.array(img_under))

        img_under, img_over, img_label = self.trans(img_under, img_over, img_label)

        return {'under': img_under, 'over': img_over, 'label': img_label, 'label_path': self.label_path[index]}

    def __len__(self):
        return len(self.scenes_list)
