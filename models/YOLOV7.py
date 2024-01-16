import torch
import torch.nn as nn

from models.common import BasicConv, MP, ELAN, ELANW, SPPCSPC, RepConv, ImplicitA, ImplicitM


class YOLOV7(nn.Module):
    def __init__(self, B=3, C=20, strides=(8,16,32), deploy=False):
        super(YOLOV7, self).__init__()
        self.strides = strides
        self.deploy = deploy
        self.B, self.C = B, C
        in_channels = 3
        yolo_channels = (5 + C) * B

        # backbone
        self.conv1 = BasicConv(in_channels, 32, 3, 1)
        self.conv2 = BasicConv(32, 64, 3, 2)
        self.conv3 = BasicConv(64, 64, 3, 1)
        self.conv4 = BasicConv(64, 128, 3, 2)

        self.elan1 = ELAN(128, 256)
        self.mp1 = MP(256, 128)
        self.elan2 = ELAN(256, 512)
        self.mp2 = MP(512, 256)
        self.elan3 = ELAN(512, 1024)
        self.mp3 = MP(1024, 512)
        self.elan4 = ELAN(1024, 1024)

        # head
        self.sppcspc = SPPCSPC(1024, 512)

        self.conv5 = BasicConv(512, 256, 1, 1)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv6 = BasicConv(1024, 256, 1, 1)

        self.elanw1 = ELANW(512, 256)

        self.conv7 = BasicConv(256, 128, 1, 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv8 = BasicConv(512, 128, 1, 1)

        self.elanw2 = ELANW(256, 128)
        self.mp4 = MP(128, 128)
        self.elanw3 = ELANW(512, 256)
        self.mp5 = MP(256, 256)
        self.elanw4 = ELANW(1024, 512)

        self.repconv1 = RepConv(128, 256, 3, 1)
        self.repconv2 = RepConv(256, 512, 3, 1)
        self.repconv3 = RepConv(512, 1024, 3, 1)

        self.yolo1 = nn.Conv2d(256, yolo_channels, 1)
        self.yolo2 = nn.Conv2d(512, yolo_channels, 1)
        self.yolo3 = nn.Conv2d(1024, yolo_channels, 1)

        self.ia = nn.ModuleList(ImplicitA(ch) for ch in (256, 512, 1024))
        self.im = nn.ModuleList(ImplicitM(yolo_channels) for _ in (256, 512, 1024))

    def forward(self, x):
        # backbone
        x = self.conv4(self.conv3(self.conv2(self.conv1(x))))

        x = self.elan1(x)
        x_24 = self.elan2(torch.cat(self.mp1(x), dim=1))
        x_37 = self.elan3(torch.cat(self.mp2(x_24), dim=1))
        x = self.elan4(torch.cat(self.mp3(x_37), dim=1))

        # head
        x_51 = self.sppcspc(x)

        x = self.up1(self.conv5(x_51))
        x_37 = self.conv6(x_37)
        x = torch.cat([x_37, x], dim=1)

        x_63 = self.elanw1(x)

        x = self.up2(self.conv7(x_63))
        x_24 = self.conv8(x_24)
        x = torch.cat([x_24, x], dim=1)

        x_75 = self.elanw2(x)
        x_88 = self.elanw3(torch.cat(self.mp4(x_75) + [x_63], dim=1))
        x_101 = self.elanw4(torch.cat(self.mp5(x_88) + [x_51], dim=1))

        x_75 = self.repconv1(x_75)
        x_88 = self.repconv2(x_88)
        x_101 = self.repconv3(x_101)

        if self.deploy:
            yolo1 = self.yolo1(x_75)
            yolo2 = self.yolo2(x_88)
            yolo3 = self.yolo3(x_101)
        else:
            yolo1 = self.im[0](self.yolo1(self.ia[0](x_75)))
            yolo2 = self.im[1](self.yolo2(self.ia[1](x_88)))
            yolo3 = self.im[2](self.yolo3(self.ia[2](x_101)))

        return [yolo1, yolo2, yolo3]

    def fuse_implicit(self):
        print("fuse implicit")
        with torch.no_grad():
            # yolo1
            # fuse ImplicitA and Convolution
            c1, c2, _, _ = self.yolo1.weight.shape
            c1_, c2_, _, _ = self.ia[0].implicit.shape
            self.yolo1.bias += torch.matmul(self.yolo1.weight.reshape(c1, c2), self.ia[0].implicit.reshape(c2_, c1_)).squeeze(1)
            # fuse ImplicitM and Convolution
            c1, c2, _, _ = self.im[0].implicit.shape
            self.yolo1.bias *= self.im[0].implicit.reshape(c2)
            self.yolo1.weight *= self.im[0].implicit.transpose(0, 1)

            # yolo1
            # fuse ImplicitA and Convolution
            c1, c2, _, _ = self.yolo2.weight.shape
            c1_, c2_, _, _ = self.ia[1].implicit.shape
            self.yolo2.bias += torch.matmul(self.yolo2.weight.reshape(c1, c2), self.ia[1].implicit.reshape(c2_, c1_)).squeeze(1)
            # fuse ImplicitM and Convolution
            c1, c2, _, _ = self.im[1].implicit.shape
            self.yolo2.bias *= self.im[1].implicit.reshape(c2)
            self.yolo2.weight *= self.im[1].implicit.transpose(0, 1)

            # yolo1
            # fuse ImplicitA and Convolution
            c1, c2, _, _ = self.yolo3.weight.shape
            c1_, c2_, _, _ = self.ia[2].implicit.shape
            self.yolo3.bias += torch.matmul(self.yolo3.weight.reshape(c1, c2), self.ia[2].implicit.reshape(c2_, c1_)).squeeze(1)
            # fuse ImplicitM and Convolution
            c1, c2, _, _ = self.im[2].implicit.shape
            self.yolo3.bias *= self.im[2].implicit.reshape(c2)
            self.yolo3.weight *= self.im[2].implicit.transpose(0, 1)

