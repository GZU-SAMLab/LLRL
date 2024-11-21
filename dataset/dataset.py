import cv2  # 导入OpenCV库，用于图像处理
import os  # 导入操作系统相关的库，用于文件和目录操作
from os.path import join as osp  # 从os.path导入join函数，并重命名为osp，方便路径拼接
import numpy  # 导入NumPy库，用于数值计算和数组操作
import torch.utils.data  # 导入PyTorch数据工具，用于创建数据集

# 定义一个继承自torch.utils.data.Dataset的自定义数据集类
class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_root='data/', mode='train', transform=None):
        # 初始化方法，file_root为数据根目录，mode为数据模式（如训练或测试），transform为数据预处理方法

        # 列出指定目录下所有文件名，假设目录结构为data/mode/A
        self.file_list = os.listdir(osp(file_root, mode, 'A'))

        # 生成预图像文件的完整路径列表（无后缀）
        self.pre_images = [osp(file_root, mode, 'A', x) for x in self.file_list]
        # 生成后图像文件的完整路径列表（无后缀）
        self.post_images = [osp(file_root, mode, 'B', x.replace('.png', '_GT.png')) for x in self.file_list]
        # 生成标签文件的完整路径列表（无后缀）
        self.gts = [osp(file_root, mode, 'label', x.replace('.png', '_mask.png')) for x in self.file_list]

        # 保存数据预处理方法
        self.transform = transform

    def __len__(self):
        # 返回数据集的大小，即预图像文件的数量
        return len(self.pre_images)

    def __getitem__(self, idx):
        # 获取指定索引的样本

        # 获取预图像、标签和后图像的文件路径
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]

        # 读取预图像文件
        pre_image = cv2.imread(pre_image_name)
        # 以灰度模式读取标签文件
        label = cv2.imread(label_name, 0)
        # 读取后图像文件
        post_image = cv2.imread(post_image_name)

        # 检查图像是否成功读取
        if pre_image is None:
            raise ValueError(f"Error reading pre_image: {pre_image_name}")
        if post_image is None:
            raise ValueError(f"Error reading post_image: {post_image_name}")
        if label is None:
            raise ValueError(f"Error reading label: {label_name}")

        # 在通道维度上拼接预图像和后图像
        img = numpy.concatenate((pre_image, post_image), axis=2)

        # 如果有预处理方法，应用预处理方法
        if self.transform:
            [img, label] = self.transform(img, label)

        # 返回拼接后的图像和标签
        return img, label

    def get_img_info(self, idx):
        # 获取指定索引图像的信息

        # 读取预图像文件
        img = cv2.imread(self.pre_images[idx])
        if img is None:
            raise ValueError(f"Error reading image: {self.pre_images[idx]}")
        # 返回图像的高度和宽度信息
        return {"height": img.shape[0], "width": img.shape[1]}

