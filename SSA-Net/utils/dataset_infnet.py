import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from imageio import imread
from skimage import transform as sktsf
import cv2


def normalization(data):
    max = np.max(data)
    min = np.min(data)
    norm_data = (data - min) / (max - min)
    # print(min, max)
    return norm_data


class DataLoader_INF(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.nii'))
        self.labels_path = glob.glob(os.path.join(data_path, 'mask/*.nii'))
        self.edges_path = glob.glob(os.path.join(data_path, 'edge/*.nii'))

        # self.imgs_path = 'data/fold1/npy_data/imgs_train.npy'
        # self.labels_path = 'data/fold1/npy_data/imgs_mask_train.npy'

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):

        image_path = self.imgs_path[index]
        label_path = self.labels_path[index]
        edge_path = self.edges_path[index]

        image = imread(image_path, as_gray=True)
        # image = sktsf.resize(image, (512, 512), mode='reflect', anti_aliasing=False)
        # 归一化
        img = normalization(image)
        image = img.reshape(1, img.shape[0], img.shape[1])

        # label = (nib.load(label_path)).get_fdata()
        label = imread(label_path, as_gray=True)
        # label = normalization(label)
        # print("label", label.shape)
        label = label.reshape(1, label.shape[0], label.shape[1])

        edge = imread(edge_path, as_gray=True)
        # label = normalization(label)
        # print("label", label.shape)
        edge = edge.reshape(1, edge.shape[0], edge.shape[1])

        return torch.from_numpy(image), torch.from_numpy(label), torch.from_numpy(edge)

        # image = np.load(self.imgs_path)
        # label = np.load(self.labels_path)
        # # print("index:", index)
        # image = image[index]
        # image = normalization(image)
        # label = label[index]
        # return torch.from_numpy(image), torch.from_numpy(label)



    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)
        # image = np.load(self.imgs_path)
        # return image.shape[0]