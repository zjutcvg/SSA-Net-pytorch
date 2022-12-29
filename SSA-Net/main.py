from model.resSCNN_sad import resSCNN_sad
from utils.dataset import DataLoader, PersoLoader
from utils.dataset_infnet import DataLoader_INF
from loss import *
from tensorboardX import SummaryWriter
from torch import optim
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

train_diceloss = []
val_loss = []
train_asdloss = []
accumulation_steps = 24

def joint_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating mis-alignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train_net(net, device, data_path, val_path, epochs=400, batch_size=2, lr=1e-3):
    # 加载训练集
    dataset = DataLoader(data_path, mode_flag=False)
    valdata = DataLoader(val_path, mode_flag=False)
    # dataset_infnet = DataLoader_INF(data_path)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=valdata, batch_size=batch_size, shuffle=False)
    # train_loader_inf = torch.utils.data.DataLoader(dataset=dataset_infnet, batch_size=batch_size, shuffle=True)

    # 定义Loss算法
    # dice loss
    diceloss = DiceLoss().to(device)
    # cnd = CrossentropyND
    # asdloss = ASD().to(device)

    BCE = nn.BCEWithLogitsLoss() # 单类
    CE = nn.CrossEntropyLoss()  # 多分类
    # best_loss统计，初始化为正无穷
    loss = float('inf')
    writer = SummaryWriter('runs/fold0/exp_s2lnet/train')
    writer2 = SummaryWriter('runs/fold0/exp_s2lnet/val')
    trainloss = 0
    # 训练epochs次
    for epoch in range(epochs):
        optimizer = torch.optim.Adam(net.parameters(), lr=lr * (0.1 ** (epoch // 100)), weight_decay=1e-8, )
        # ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
        print('-' * 30)
        # print('Epoch {}/{}'.format(epoch, epochs - 1))
        # print('-' * 30)
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        # """
        for i, (image, label) in enumerate(train_loader):
            # 将数据拷贝到device中
            # label = torch.tensor(label, dtype=torch.float32)
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # print(i)

            # -------使用网络参数，输出预测结果-----
            # # 计算loss
            pred, loss_distill = net(image, True)
            loss_seg = diceloss(pred, label)
            loss = loss_seg * 1.0
            loss += loss_distill * 0.1

            train_diceloss.append(loss.item())
            loss = loss / accumulation_steps
            loss.backward()
            if ((i + 1) % accumulation_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
        # print(train_loss, len(train_loss))
        print('Train Epoch: {}/{} ,  Loss:{:.6f}'.format(epoch, epochs-1,
                                                         sum(train_diceloss[0:len(train_diceloss)])/len(train_diceloss)))
        loss_out = sum(train_diceloss[0:len(train_diceloss)]) / len(train_diceloss)
        writer.add_scalar('Dice Loss', loss_out, global_step=epoch)
        if ((epoch+1) % 10 == 0) and ((epoch+1) >= 50):
            torch.save(net.state_dict(), 'checkpoints/weakly/fold0/t1/weight_{}.pth'.format(epoch+1))

        # ----------infnet_loss--------------

        net.eval()
        with torch.no_grad():
            for i, (image, label) in enumerate(val_loader):
                # label = torch.tensor(label, dtype=torch.float32)
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # image = Variable(image).to(device=device, dtype=torch.float32)
                # label = Variable(label).to(device=device, dtype=torch.float32)
                # --------prediction---------
                # pred = net(image)
                # pred_5, pred_4, pred_3, pred_2, edge = net(image)
                # pred = F.sigmoid(pred_2)
                # ---------prediction_SAD--------
                pred, _ = net(image)

                valloss = diceloss(pred, label)
                val_loss.append(valloss.item())

            print('Val Epoch: {}/{} ,  val_loss:{:.6f}'.format(epoch, epochs - 1,
                                                          sum(val_loss[0:len(val_loss)]) / len(val_loss)))
            valloss_out = sum(val_loss[0:len(val_loss)]) / len(val_loss)
            writer2.add_scalar('Dice Loss', valloss_out, global_step=epoch)
            # ExpLR.step()
            print("lr=", optimizer.state_dict()['param_groups'][0]['lr'])



if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络
    net = resSCNN_sad(1, 1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "data/weakly/fold0/t1/train"
    val_path = "data/weakly/fold0/train"
    train_net(net, device, data_path, val_path)
