import glob
import numpy as np
import torch
import os
import cv2
from model.unet import UNet
from model.Unet11 import UNet11
from model.CE_Net import CE_Net
from model.ResUnetSCNN import resSCNN
from model.resUnet import resUnet
from model.nnUnet import nnUNet
from model.InfNet_Res2Net import Inf_Net
from model.resSCNNp1 import resSCNNp1
from model.resSCNN_sad import resSCNN_sad
from skimage import transform as sktsf
from imageio import imread, imwrite
from utils.dataset import *
from loss import DiceLoss
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import nibabel as nib

"""
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(1, 1)
    # MultiUnet
    # net = MultiResUnet(1, 1, dim='2d')

    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数

    net.load_state_dict(torch.load('checkpoints/unet_100/weight.pth', map_location=device))


    # 测试模式
    net.eval()
    # 读取所有图片路径
    # tests_path = glob.glob('data/test/*.nii')
    tests_path = glob.glob('data100/test/*.nii')
    # lungs_path = glob.glob('data/lung_mask_test/*.nii')
    # 遍历所有图片
    test_loss = []
    testloss = 0
    diceloss = DiceLoss().to(device)
    with torch.no_grad():
        for test_path in tests_path:
            # 保存结果地址
            save_res_path = test_path.split('.')[0] + '_unet.png'
            # 读取图片
            img = imread(test_path, as_gray=True)
            image = sktsf.resize(img, (512, 512), mode='reflect', anti_aliasing=False)
            img = normalization(image)
            # lung = (nib.load(lung_path)).get_fdata()
            # img = img * lung

            # 转为batch为1，通道为1，大小为512*512的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            # 转为tensor
            img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # 预测
            pred = net(img_tensor)
            loss = diceloss(pred, pred)

            test_loss.append(loss)
            print(loss)
            testloss += loss
            # 提取结果
            pred = np.array(pred.data.cpu()[0])[0]
            # 处理结果
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            # 保存图片
            cv2.imwrite(save_res_path, pred)
        testloss = testloss / len(tests_path)
        print('Test Loss:', testloss)
"""

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    # net = UNet(1, 1)
    # net = MultiResUnet(1, 1, dim='2d')
    # net = UNet11(1, 1)
    # net = CE_Net(1, 1)
    # net = resSCNN(1, 1)
    net = resSCNN_sad(1, 1)
    # net = resUnet(1, 1)
    # net = nnUNet(1, 1)
    # net = Inf_Net(1, 1)
    # net = resSCNNp1(1,1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数

    net.load_state_dict(torch.load('../checkpoints100/resSCNN_SAD3/weight_150.pth', map_location=device))
    # writer_test = SummaryWriter('runs100/exp1_resSCNN/test')
    # 测试模式
    net.eval()
    # 读取所有图片路径
    # tests_path = glob.glob('data/test/*.nii')
    tests_path = "../data100_INF/test/img/70.nii"
    dataset = DataLoader(tests_path)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=1)
    # lungs_path = glob.glob('data/lung_mask_test/*.nii')
    # 遍历所有图片
    test_loss = []
    testloss = 0
    diceloss = DiceLoss().to(device)
    i = 0
    with torch.no_grad():
        img_raw = imread(tests_path, as_gray=True)
        img = normalization(img_raw)
        image = img.reshape(1, 1, img.shape[0], img.shape[1])
        image = torch.from_numpy(image)
        # 保存结果地址
        save_res_path = "../data100_INF/ttt"
        # 读取图片
        # """
        image = image.to(device=device, dtype=torch.float32)

        # 预测
        x = net.firstconv(image)
        layer1 = net.layer1(x)
        layer2 = net.layer2(layer1)
        layer3 = net.layer3(layer2)
        layer4 = net.layer4(layer3)

        pred1 = net.at_gen_upsample(layer4)
        pred1 = net.at_gen_upsample(pred1)
        pred1 = net.at_gen_upsample(pred1)
        pred1 = net.at_gen_upsample(pred1)
        pred1 = net.at_gen_upsample(pred1)
        pred2 = np.array(pred1.data.cpu()[0])[0]
        print(pred2.shape)
        # pred = np.squeeze(pred)
        # pred2 = pred2 * 255

        pmin = np.min(pred2)
        pmax = np.max(pred2)
        pred = ((pred2-pmin)/(pmax-pmin+0.000001))*255
        pred = pred.astype(np.uint8)
        pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        print(pred.shape)

        img=img*255
        img = img.reshape(img.shape[0], img.shape[1], 1)
        print(img.shape)
        img2 = img + pred*0.9
        cv2.imwrite(os.path.join(save_res_path, '4.png'.format(i)), img2)

            # pred = net(image)
            # loss = diceloss(label, pred)
            # test_loss.append(loss)
            # print(i+1, ": ", loss.item())
            # testloss += loss.item()
            # # 提取结果
            # # test = np.array(image.data.cpu()[0])[0]
            # pred = np.array(pred.data.cpu()[0])[0]
            # # print(type(pred), pred)
            # # 处理结果
            # pred[pred >= 0.5] = 255
            # pred[pred < 0.5] = 0
            #
            # # 保存图片
            # # cv2.imwrite(os.path.join(save_res_path, '{:03d}.png'.format(i)), pred)
            # # pred = np.squeeze(pred)
            # # imwrite(os.path.join(save_res_path, '{:03d}.nii'.format(i)), pred)
            # i += 1
            # writer_test.add_scalar('Test Loss', loss, global_step=i)
        # testloss = testloss / i
        # print('Test Loss:', testloss, i)

