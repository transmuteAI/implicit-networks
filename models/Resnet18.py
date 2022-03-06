import torch
from .utils import HomogeneousChannelsMaxPool,HeterogeneousChannelsMaxPool
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool2d

class wide_basic(nn.Module):
    def __init__(self, in_planes, inner_planes, out_planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=1, bias=True)
        #self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(inner_planes)
        self.conv2 = nn.Conv2d(inner_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True),)

    def forward(self, x):
        x_n = F.relu(self.bn1(x))
        out = F.relu(self.bn2(self.conv1(x_n))) #self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(out)
        if self.shortcut is not None:
            out += self.shortcut(x_n)
        else:
            out += x
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes, ratio=None, mplc_kernel_size=None, dropout_rate=0, deltaorth=False, initial_stride=1, fixparams=True):
        super(ResNet18, self).__init__()
        if ratio:
            if sum(ratio)==11:
                c1 = 88
            elif sum(ratio)==43:
                c1 = 129
        else:
            if mplc_kernel_size==4:
                c1 = 34*4
            elif mplc_kernel_size==8:
                c1 = 24*8
        nStages = [c1, c1, c1*2, c1*4, c1*8]
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.in_planes = nStages[0]
        self.layer1 = self._wide_layer(wide_basic, nStages[1], 2, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], 2, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], 2, dropout_rate, stride=2)
        self.layer4 = self._wide_layer(wide_basic, nStages[4], 2, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[4], momentum=0.9)
        self.avgpool = AdaptiveAvgPool2d((1,1))
        if ratio:
            self.mplc = HeterogeneousChannelsMaxPool(ratio,mplc_kernel_size)
            k4 = nStages[4]//sum(ratio)
            out_size = sum([(k4*ratio[i])//mplc_kernel_size[i] for i in range(len(ratio))])
        else:
            self.mplc = HomogeneousChannelsMaxPool(mplc_kernel_size)
            out_size = nStages[4]//mplc_kernel_size

        self.linear = nn.Linear(out_size, num_classes)


    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []
        inner_planes = planes
        for stride in strides:
            out_planes = planes
            layers.append(block(self.in_planes, inner_planes, out_planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out1 = self.mplc(out)
        out = self.layer2(out)
        out2 = self.mplc(out)
        out = self.layer3(out)
        out3 = self.mplc(out)
        out = self.layer4(out)
        out = F.relu(self.bn1(out))
        out = self.mplc(out)
        out4 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out,[out1,out2,out3,out4]
