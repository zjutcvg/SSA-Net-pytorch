import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from loss import DiceLoss
# from torchsummary import summary
from torch.nn.parameter import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nonlinearity = partial(F.relu, inplace=True)


class SpatialSoftmax(nn.Module):
    def __init__(self, temperature=1, device='cpu'):
        super(SpatialSoftmax, self).__init__()

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature).to(device)
        else:
            self.temperature = 1.

    def forward(self, feature):
        feature = feature.view(feature.shape[0], -1, feature.shape[1] * feature.shape[2])
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)

        return softmax_attention


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResBlock, self).__init__()

        self.Res = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch)

        )

        self.right = shortcut

    def forward(self, x):
        x1 = x if self.right is None else self.right(x)
        x2 = self.Res(x)
        out = x1 + x2
        return F.relu(out)


class resSCNN_sad(nn.Module):
    def __init__(self, in_ch=1, num_classes=1):
        super(resSCNN_sad, self).__init__()

        filters = [64, 128, 256, 512]
        self.scale_sad_distill = 0.1
        self.bce_loss = nn.BCELoss()

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)

        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(inchannel=64, outchannel=64, bloch_num=3)
        self.layer2 = self._make_layer(inchannel=64, outchannel=128, bloch_num=4, stride=2)
        self.layer3 = self._make_layer(inchannel=128, outchannel=256, bloch_num=6, stride=2)
        self.layer4 = self._make_layer(inchannel=256, outchannel=512, bloch_num=3, stride=2)

        self.message_passing = nn.ModuleList()
        self.message_passing.add_module('up_down', nn.Conv2d(512, 512, (1, 9), padding=(0, 9 // 2), bias=False))
        self.message_passing.add_module('down_up', nn.Conv2d(512, 512, (1, 9), padding=(0, 9 // 2), bias=False))
        self.message_passing.add_module('left_right',
                                        nn.Conv2d(512, 512, (9, 1), padding=(9 // 2, 0), bias=False))
        self.message_passing.add_module('right_left',
                                        nn.Conv2d(512, 512, (9, 1), padding=(9 // 2, 0), bias=False))

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.at_gen_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.at_gen_l2_loss = nn.MSELoss(reduction='mean')
        # self.at_gen_l2_loss = DiceLoss().to(device)

    def _make_layer(self, inchannel, outchannel, bloch_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, bloch_num):
            layers.append(ResBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def at_gen(self, x1, x2):
        """
        x1 - previous encoder step feature map
        x2 - current encoder step feature map
        """

        # G^2_sum
        sps = SpatialSoftmax(device=x1.device)

        if x1.size() != x2.size():
            x1 = torch.sum(x1 * x1, dim=1)
            # print(x1.shape)
            x1 = sps(x1)
            x2 = torch.sum(x2 * x2, dim=1, keepdim=True)
            x2 = torch.squeeze(self.at_gen_upsample(x2), dim=1)
            # print(x2.shape)
            x2 = sps(x2)
        else:
            x1 = torch.sum(x1 * x1, dim=1)
            x1 = sps(x1)
            x2 = torch.sum(x2 * x2, dim=1)
            x2 = sps(x2)
        # print(x2.shape)

        loss = self.at_gen_l2_loss(x1, x2)
        # print(loss)
        return loss

    def forward(self, x, sad_loss=False):
        # Encoder
        x = self.firstconv(x)  # 1,1,512,512 ---->1,64,128,128
        e1 = self.layer1(x)  # 1,64,128,128 --->1,64,128,128

        e2 = self.layer2(e1)  # 1,64,128,128 --->1,128,64,64
        if sad_loss:
            loss_2 = self.at_gen(e1, e2)

        e3 = self.layer3(e2)  # 1,128,64,64----->1,256,32,32
        if sad_loss:
            loss_3 = self.at_gen(e2, e3)

        e4 = self.layer4(e3)  # 1,256,32,32---->1,512,16,16
        if sad_loss:
            loss_4 = self.at_gen(e3, e4)

        # e_34 = torch.cat((e3, 34), dim=1)

        e4 = self.message_passing_forward(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        seg_pred = F.sigmoid(out)

        if sad_loss:
            loss_distill = loss_2 + loss_3 + loss_4
        else:
            loss_distill = 0

        return seg_pred, loss_distill

    def message_passing_forward(self, x):
        Vertical = [True, True, False, False]
        Reverse = [False, True, False, True]
        for ms_conv, v, r in zip(self.message_passing, Vertical, Reverse):
            x = self.message_passing_once(x, ms_conv, v, r)
        return x

    def message_passing_once(self, x, conv, vertical=True, reverse=False):
        """
        Argument:
        ----------
        x: input tensor
        vertical: vertical message passing or horizontal
        reverse: False for up-down or left-right, True for down-up or right-left
        """
        nB, C, H, W = x.shape
        if vertical:
            slices = [x[:, :, i:(i + 1), :] for i in range(H)]
            dim = 2
        else:
            slices = [x[:, :, :, i:(i + 1)] for i in range(W)]
            dim = 3
        if reverse:
            slices = slices[::-1]

        out = [slices[0]]
        for i in range(1, len(slices)):
            out.append(slices[i] + F.relu(conv(out[i - 1])))
        if reverse:
            out = out[::-1]
        return torch.cat(out, dim=dim)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = resSCNN(1,1).to(device)
#
# summary(model, (1,512,512))