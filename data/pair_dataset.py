import os.path
import torchvision.transforms.v2 as transforms
from data.base_dataset import BaseDataset
from PIL import Image
import random
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_img(img, downsample):
    # PyTorch: [C, H, W]
    img = img.numpy()
    # NumPy/OpenCV: [H, W, C]
    img = np.transpose(img, (1, 2, 0))
    # resize(width, height)
    img_low = cv2.resize(img, (int(img.shape[1] * downsample), int(img.shape[0] * downsample)))
    img_low = torch.from_numpy(img_low).permute(2, 0, 1)
    return img_low

class PairDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.shortSide, self.longSide = [int(i) for i in opt.fineSize.split(',')]
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.under_path) # might be under exposure dir
        print(f"Using {self.dir_A} as under exposure dir")
        self.dir_B = os.path.join(opt.dataroot, opt.over_path) # might be over exposure dir
        self.dir_C = os.path.join(opt.dataroot, opt.label_path) # might be ground truth dir

        # medium exposure
        self.dir_M = os.path.join(opt.dataroot, opt.medium_path)
        print(f"Using {self.dir_M} as medium exposure dir")
        print(f"Using {self.dir_B} as over exposure dir")
        print(f"Using {self.dir_C} as ground truth dir")

        self.A_paths = []
        self.B_paths = []
        self.C_paths = []

        # medium exposure
        self.M_paths = []

        for i in range(1, len(os.listdir(self.dir_A)) + 1):
            self.A_paths.append(self.dir_A + "/" + str(i) + ".jpg")
            self.B_paths.append(self.dir_B + "/" + str(i) + ".jpg")
            self.C_paths.append(self.dir_C + "/" + str(i) + ".jpg")

            # medium exposure
            self.M_paths.append(self.dir_M + "/" + str(i) + ".jpg")


        self.image_num = len(self.A_paths)


        # self.trans_h = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),   # 随机水平翻转
        #     transforms.RandomVerticalFlip(),     # 随机垂直翻转
        #     transforms.RandomCrop((self.longSide, self.shortSide)),    # 随机裁剪到指定大小
        #     transforms.ToTensor()              # 将PIL图像转换为张量
        # ])

        self.trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),   # 随机水平翻转
            transforms.RandomVerticalFlip(),     # 随机垂直翻转
            transforms.RandomCrop((self.shortSide, self.longSide)),    # 随机裁剪到指定大小
            transforms.ToTensor()               # 将PIL图像转换为张量
        ])


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.image_num] # a single image file
        B_path = self.B_paths[index % self.image_num]
        C_path = self.C_paths[index % self.image_num]

        # medium exposure
        M_path = self.M_paths[index % self.image_num]


        # Image.open() returns an image object and the convert() function converts the image to RGB
        # the shape should be (channel, height, width)
        A_img = Image.open(A_path).convert('RGB')
        # a = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # b = Image.open(A_path.replace(
        #     "low", "high").replace("A", "B")).convert('RGB')
        C_img = Image.open(C_path).convert('RGB')

        # medium exposure
        M_img = Image.open(M_path).convert('RGB')
        # m = Image.open(M_path).convert('RGB')

        w, h = A_img.size

        if (h > w):
            A_img = A_img.rotate(-90, expand=True)
            B_img = B_img.rotate(-90, expand=True)
            C_img = C_img.rotate(-90, expand=True)
            M_img = M_img.rotate(-90, expand=True)
        A_img, B_img, C_img, M_img = self.trans(A_img, B_img, C_img, M_img)


        # plt.figure(figsize=(10, 10))

        # print(A_img.shape)

        # plt.subplot(1, 4, 1)
        # plt.imshow(A_img.permute(1, 2, 0).detach().cpu().numpy())
        # plt.title('Under')

        # plt.subplot(1, 4, 3)
        # plt.imshow(M_img.permute(1, 2, 0).detach().cpu().numpy())
        # plt.title('Medium')

        # plt.subplot(1, 4, 2)
        # plt.imshow(B_img.permute(1, 2, 0).detach().cpu().numpy())
        # plt.title('Over')

        # plt.subplot(1, 4, 4)
        # plt.imshow(C_img.permute(1, 2, 0).detach().cpu().numpy())
        # plt.title('C_img')

        # plt.show()

        A_img_low = process_img(A_img, self.opt.downsample)
        B_img_low = process_img(B_img, self.opt.downsample)
        M_img_low = process_img(M_img, self.opt.downsample)
        batch = {'A_low': A_img_low, 'B_low': B_img_low, 'C': C_img, 'M_low': M_img_low, 'M': M_img}
        return batch

    def __len__(self):
        return self.image_num

    def name(self):
        return 'PairDataset'
