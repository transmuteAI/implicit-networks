import torch
from .utils import HomogeneousChannelsMaxPool, HeterogeneousChannelsMaxPool
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, ELU

class CNNBasic(torch.nn.Module):

    def __init__(self, n_classes, ratio=None, mplc_kernel_size=None):
        super(CNNBasic, self).__init__()
        if ratio:
            if sum(ratio)==11:
                c1 = 88
        else:
            c1 = 128
        c2, c3, c4, c5 = int(1.5*c1), 2*c1, 3*c1, 4*c1
        self.conv1 = Conv2d(1, c1, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1), bias=False)
        self.btn1 = BatchNorm2d(c1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(c1, c2, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.btn2 = BatchNorm2d(c2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv2d(c2, c3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.btn3 = BatchNorm2d(c3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = Conv2d(c3, c3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.btn4 = BatchNorm2d(c3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = Conv2d(c3, c4, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.btn5 = BatchNorm2d(c4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = Conv2d(c4, c5, kernel_size=(5, 5), stride=(1, 1), bias=False)
        self.btn6 = BatchNorm2d(c5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mpl = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False)
        self.avgpl = AdaptiveAvgPool2d(1)
        self.elu = ELU(alpha=1.0,inplace=True)
        if ratio:
            self.mplc = HeterogeneousChannelsMaxPool(ratio,mplc_kernel_size)
        else:
            self.mplc = HomogeneousChannelsMaxPool(mplc_kernel_size)
        self.fullnet = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(alpha=1,inplace=True),
            torch.nn.Linear(128, n_classes),
        )

    def forward(self, x):
        lst=[]

        x = self.elu(self.btn1(self.conv1(x)))
        y = self.mplc(x)
        lst.append(y)
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
        x = self.avgpl(y)

        x = self.fullnet(x.reshape(x.shape[0], -1))

        return x,lst
