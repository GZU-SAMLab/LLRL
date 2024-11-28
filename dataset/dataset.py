import cv2  # 导入OpenCV库，用于图像处理
import os  # 导入操作系统相关的库，用于文件和目录操作
from os.path import join as osp  # 从os.path导入join函数，并重命名为osp，方便路径拼接
import numpy  # 导入NumPy库，用于数值计算和数组操作
import torch.utils.data  # 导入PyTorch数据工具，用于创建数据集

# 定义一个继承自torch.utils.data.Dataset的自定义数据集类
class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_root='data/', mode='train', transform=None):
        # Initialization method. file_root is the root directory of the data, mode is the data mode (e.g., train or test),
        # and transform is the data preprocessing method.

        # List all filenames in the specified directory, assuming directory structure is data/mode/A
        self.file_list = os.listdir(osp(file_root, mode, 'A'))

        # Generate a list of full paths for the pre-images (without extension)
        self.pre_images = [osp(file_root, mode, 'A', x) for x in self.file_list]
        # Generate a list of full paths for the post-images (without extension)
        self.post_images = [osp(file_root, mode, 'B', x.replace('.png', '_GT.png')) for x in self.file_list]
        # Generate a list of full paths for the label files (without extension)
        self.gts = [osp(file_root, mode, 'label', x.replace('.png', '_mask.png')) for x in self.file_list]

        # Save the data preprocessing method
        self.transform = transform

    def __len__(self):
        # Return the size of the dataset, which is the number of pre-images
        return len(self.pre_images)

    def __getitem__(self, idx):
        # Get the sample at the specified index

        # Get the file paths for the pre-image, label, and post-image
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]

        # Read the pre-image file
        pre_image = cv2.imread(pre_image_name)
        # Read the label file in grayscale mode
        label = cv2.imread(label_name, 0)
        # Read the post-image file
        post_image = cv2.imread(post_image_name)

        # Check if the images were successfully read
        if pre_image is None:
            raise ValueError(f"Error reading pre_image: {pre_image_name}")
        if post_image is None:
            raise ValueError(f"Error reading post_image: {post_image_name}")
        if label is None:
            raise ValueError(f"Error reading label: {label_name}")

        # Concatenate the pre-image and post-image along the channel dimension
        img = numpy.concatenate((pre_image, post_image), axis=2)

        # If a transform method is provided, apply it to the image and label
        if self.transform:
            [img, label] = self.transform(img, label)

        # Return the concatenated image and label
        return img, label

    def get_img_info(self, idx):
        # Get the information of the image at the specified index

        # Read the pre-image file
        img = cv2.imread(self.pre_images[idx])
        if img is None:
            raise ValueError(f"Error reading image: {self.pre_images[idx]}")
        # Return the height and width of the image
        return {"height": img.shape[0], "width": img.shape[1]}
