import cv2  
import os  
from os.path import join as osp  
import numpy  
import torch.utils.data  

# 定义一个继承自torch.utils.data.Dataset的自定义数据集类
class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_root='data/', mode='train', transform=None):
        
        self.file_list = os.listdir(osp(file_root, mode, 'A'))

        
        self.pre_images = [osp(file_root, mode, 'A', x) for x in self.file_list]
        
        self.post_images = [osp(file_root, mode, 'B', x.replace('.png', '_GT.png')) for x in self.file_list]
        
        self.gts = [osp(file_root, mode, 'label', x.replace('.png', '_mask.png')) for x in self.file_list]

        
        self.transform = transform

    def __len__(self):
        
        return len(self.pre_images)

    def __getitem__(self, idx):
        

       
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]

        
        pre_image = cv2.imread(pre_image_name)
       
        label = cv2.imread(label_name, 0)
        
        post_image = cv2.imread(post_image_name)

        
        if pre_image is None:
            raise ValueError(f"Error reading pre_image: {pre_image_name}")
        if post_image is None:
            raise ValueError(f"Error reading post_image: {post_image_name}")
        if label is None:
            raise ValueError(f"Error reading label: {label_name}")

        
        img = numpy.concatenate((pre_image, post_image), axis=2)

        
        if self.transform:
            [img, label] = self.transform(img, label)

        
        return img, label

    def get_img_info(self, idx):
        

        
        img = cv2.imread(self.pre_images[idx])
        if img is None:
            raise ValueError(f"Error reading image: {self.pre_images[idx]}")
        
        return {"height": img.shape[0], "width": img.shape[1]}

