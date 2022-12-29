import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import random
from imageio import imread
from skimage import transform as sktsf
import cv2


def normalization(data):
    max = np.max(data)
    min = np.min(data)
    norm_data = (data - min) / (max - min)
    # print(min, max)
    return norm_data


class DataLoader(Dataset):
    def __init__(self, data_path, mode_flag=False):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        # mode-true=train, mode-false=val
        self.mode_flag = mode_flag
        self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.nii'))
        self.labels_path = glob.glob(os.path.join(data_path, 'mask/*.nii'))

        # self.imgs_path = 'data/fold1/npy_data/imgs_train.npy'
        # self.labels_path = 'data/fold1/npy_data/imgs_mask_train.npy'

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):

        image_path = self.imgs_path[index]
        label_path = self.labels_path[index]

        img = imread(image_path, as_gray=True)
        # img = normalization(img)
        image = img.reshape(1, img.shape[0], img.shape[1])

        # label = (nib.load(label_path)).get_fdata()
        label = imread(label_path, as_gray=True)
        label = label.reshape(1, label.shape[0], label.shape[1])

        if self.mode_flag:
            flip_code = random.choice([-1, 0, 1, 2])
            if flip_code != 2:
                image = self.augment(image, flip_code)
                label = self.augment(label, flip_code)

        image, label = np.array(image), np.array(label)
        image = image.astype(np.float32)
        label = label.astype(np.float32)

        return torch.from_numpy(image), torch.from_numpy(label)

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)
        # image = np.load(self.imgs_path)
        # return image.shape[0]


class TestLoader(Dataset):
    def __init__(self, data_path, mode_flag=False):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        # mode-true=train, mode-false=val
        self.mode_flag = mode_flag
        self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.nii'))
        self.labels_path = glob.glob(os.path.join(data_path, 'mask/*.nii'))

        # self.imgs_path = 'data/fold1/npy_data/imgs_train.npy'
        # self.labels_path = 'data/fold1/npy_data/imgs_mask_train.npy'

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):

        image_path = self.imgs_path[index]
        label_path = self.labels_path[index]

        name = image_path.split('\\')[-1].split('.')[0]

        img = imread(image_path, as_gray=True)
        # img = normalization(img)
        image = img.reshape(1, img.shape[0], img.shape[1])

        label = imread(label_path, as_gray=True)
        label = label.reshape(1, label.shape[0], label.shape[1])

        return torch.from_numpy(image), torch.from_numpy(label), name

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


class PersoLoader(Dataset):
    def __init__(self, data_path, perso_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.perso_path = perso_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'img/*.nii'))
        self.labels_path = glob.glob(os.path.join(data_path, 'mask/*.nii'))
        self.persos_path = glob.glob(os.path.join(perso_path, 'img/*.nii'))
        self.perlabel_path = glob.glob(os.path.join(perso_path, 'mask/*.nii'))

        # self.imgs_path = 'data/fold1/npy_data/imgs_train.npy'
        # self.labels_path = 'data/fold1/npy_data/imgs_mask_train.npy'

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):

        image_path = self.imgs_path[index]
        label_path = self.labels_path[index]

        img = imread(image_path, as_gray=True)
        img = normalization(img)
        image = img.reshape(1, img.shape[0], img.shape[1])

        # label = (nib.load(label_path)).get_fdata()
        label = imread(label_path, as_gray=True)
        label = label.reshape(1, label.shape[0], label.shape[1])

        return torch.from_numpy(image), torch.from_numpy(label)

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)
        # image = np.load(self.imgs_path)
        # return image.shape[0]

# if __name__ == "__main__":
#     dataset = DataLoader("../data/fold0/train")
#     print("数据个数：", len(dataset))
#     train_loader = torch.utils.data.DataLoader(dataset=dataset,
#                                                batch_size=2,
#                                                shuffle=True)
#     for image, label in train_loader:
#         print(image.shape)
#         print(label.shape)

