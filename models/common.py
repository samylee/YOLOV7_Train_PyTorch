import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act='silu'):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act == 'silu' else nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MP, self).__init__()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = BasicConv(in_channels, out_channels, 1, 1)
        self.conv2 = BasicConv(in_channels, out_channels, 1, 1)
        self.conv3 = BasicConv(out_channels, out_channels, 3, 2)

    def forward(self, x):
        out1 = self.conv1(self.mp(x))
        out2 = self.conv3(self.conv2(x))
        return [out2, out1]


class ELAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELAN, self).__init__()
        mid_channels = out_channels // 4
        self.conv1 = BasicConv(in_channels, mid_channels, 1, 1)
        self.conv2 = BasicConv(in_channels, mid_channels, 1, 1)
        self.conv3 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv4 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv5 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv6 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv7 = BasicConv(mid_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv4(self.conv3(out2))
        out4 = self.conv6(self.conv5(out3))
        return self.conv7(torch.cat([out4, out3, out2, out1], dim=1))


class ELANW(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELANW, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = BasicConv(in_channels, out_channels, 1, 1)
        self.conv2 = BasicConv(in_channels, out_channels, 1, 1)
        self.conv3 = BasicConv(out_channels, mid_channels, 3, 1)
        self.conv4 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv5 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv6 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv7 = BasicConv(out_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        return self.conv7(torch.cat([out6, out5, out4, out3, out2, out1], dim=1))


class SPPCSPC(nn.Module):
    def __init__(self, in_channels, out_channels, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        mid_channels = int(2 * out_channels * e)
        self.conv1 = BasicConv(in_channels, mid_channels, 1, 1)
        self.conv2 = BasicConv(in_channels, mid_channels, 1, 1)
        self.conv3 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv4 = BasicConv(mid_channels, mid_channels, 1, 1)
        self.mps = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.conv5 = BasicConv(mid_channels * 4, mid_channels, 1, 1)
        self.conv6 = BasicConv(mid_channels, mid_channels, 3, 1)
        self.conv7 = BasicConv(mid_channels * 2, out_channels, 1, 1)

    def forward(self, x):
        x_tmp = self.conv4(self.conv3(self.conv1(x)))
        out1 = self.conv6(self.conv5(torch.cat([x_tmp] + [mp(x_tmp) for mp in self.mps], dim=1)))
        out2 = self.conv2(x)
        return self.conv7(torch.cat((out1, out2), dim=1))


class RepConv(nn.Module):
    # https://arxiv.org/abs/2101.03697
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, act='silu'):
        super(RepConv, self).__init__()
        self.act = nn.SiLU() if act == 'silu' else nn.LeakyReLU(0.1)
        self.rbr_identity = (nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None)
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
