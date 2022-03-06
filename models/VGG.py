import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool2d, Conv2d, BatchNorm2d, MaxPool2d, ELU
from .utils import HomogeneousChannelsMaxPool,HeterogeneousChannelsMaxPool

class VGG(torch.nn.Module):

    def __init__(self, n_classes, ratio=None, mplc_kernel_size=None):

        super(VGG, self).__init__()
        if ratio:
            if sum(ratio)==11:
                c1 = 99
            elif sum(ratio)==43:
                c1 = 129
        else:
            if mplc_kernel_size==4:
                c1 = 144
            elif mplc_kernel_size==8:
                c1 = 200
        self.chnls = [c1, 2*c1, 4*c1, 4*c1, 8*c1, 8*c1, 8*c1, 8*c1, 1024, n_classes]
        self.ksize = [3,2, 3,2, 3,3,2, 3,3,2, 3,3,2]
        self.pad = [1]*8
        self.conv1 = Conv2d(3, self.chnls[0], kernel_size=(self.ksize[0], self.ksize[0]), stride=(1, 1), padding=(self.pad[0], self.pad[0]), bias=False)
        self.btn1 = BatchNorm2d(self.chnls[0], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(self.chnls[0], self.chnls[1], kernel_size=(self.ksize[1], self.ksize[1]), stride=(1, 1), padding=(self.pad[1], self.pad[1]), bias=False)
        self.btn2 = BatchNorm2d(self.chnls[1], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv2d(self.chnls[1], self.chnls[2], kernel_size=(self.ksize[2], self.ksize[2]), stride=(1, 1), padding=(self.pad[2], self.pad[2]), bias=False)
        self.btn3 = BatchNorm2d(self.chnls[2], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = Conv2d(self.chnls[2], self.chnls[3], kernel_size=(self.ksize[3], self.ksize[3]), stride=(1, 1), padding=(self.pad[3], self.pad[3]), bias=False)
        self.btn4 = BatchNorm2d(self.chnls[3], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = Conv2d(self.chnls[3], self.chnls[4], kernel_size=(self.ksize[4], self.ksize[4]), stride=(1, 1), padding=(self.pad[4], self.pad[4]), bias=False)
        self.btn5 = BatchNorm2d(self.chnls[4], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = Conv2d(self.chnls[4], self.chnls[5], kernel_size=(self.ksize[5], self.ksize[5]), stride=(1, 1), padding=(self.pad[5], self.pad[5]), bias=False)
        self.btn6 = BatchNorm2d(self.chnls[5], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv7 = Conv2d(self.chnls[5], self.chnls[6], kernel_size=(self.ksize[6], self.ksize[6]), stride=(1, 1), padding=(self.pad[6], self.pad[6]), bias=False)
        self.btn7 = BatchNorm2d(self.chnls[6], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv8 = Conv2d(self.chnls[6], self.chnls[7], kernel_size=(self.ksize[7], self.ksize[7]), stride=(1, 1), padding=(self.pad[7], self.pad[7]), bias=False)
        self.btn8 = BatchNorm2d(self.chnls[7], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mpl = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False)
        self.avgpl = AdaptiveAvgPool2d(1)
        self.elu = ELU(alpha=1.0,inplace=True)
        if ratio:
            self.mplc = HeterogeneousChannelsMaxPool(ratio,mplc_kernel_size)
            k4 = self.chnls[7]//sum(ratio)
            out_size = sum([(k4*ratio[i])//mplc_kernel_size[i] for i in range(len(ratio))])
        else:
            self.mplc = HomogeneousChannelsMaxPool(mplc_kernel_size)
            out_size = self.chnls[7]//mplc_kernel_size
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(out_size, self.chnls[8]),
            torch.nn.BatchNorm1d(self.chnls[8]),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.chnls[8], self.chnls[9]),
        )

    def forward(self, x):
        lst=[]
        x = self.elu(self.btn1(self.conv1(x)))
        y = self.mplc(x)
        lst.append(y)
        x = self.mpl(x)

        x = self.elu(self.btn2(self.conv2(x)))
        y = self.mplc(x)
        lst.append(y)
        x = self.mpl(x)

        x = self.elu(self.btn3(self.conv3(x)))
        y = self.mplc(x)
        lst.append(y)
        x = self.elu(self.btn4(self.conv4(x)))
        y = self.mplc(x)
        lst.append(y)
        x = self.mpl(x)

        x = self.elu(self.btn5(self.conv5(x)))
        y = self.mplc(x)
        lst.append(y)
        x = self.elu(self.btn6(self.conv6(x)))
        y = self.mplc(x)
        lst.append(y)
        x = self.mpl(x)

        x = self.elu(self.btn7(self.conv7(x)))
        y = self.mplc(x)
        lst.append(y)
        x = self.elu(self.btn8(self.conv8(x)))
        y = self.mplc(x)
        lst.append(y)
        x = self.avgpl(y)

        x = self.fully_net(x.reshape(x.shape[0], -1))

        return x,lst
