import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import AdaptiveAvgPool2d
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
import torchvision.datasets as datasets
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, ELU

import matplotlib.pyplot as plt
import time, os, copy, math, sys, random, pickle
from typing import List, Tuple, Any, Union
import numpy as np
from PIL import Image
from tqdm import tqdm
from itertools import accumulate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class MaxPoolChannels(torch.nn.Module):   
    def __init__(self):
        super(MaxPoolChannels, self).__init__()
        self.ratio = [32,8,2,1]
        self.sum_ratio = sum(self.ratio)
        self.kernel_size = [8,4,2,1]
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        b = input.shape[0]
        c = input.shape[1]
        self.partition = [(c//self.sum_ratio)*i for i in self.ratio]
        s = input.shape[2:]
        groups = list(input.tensor_split(list(accumulate(self.partition))[:-1],1))
        for i in range(len(groups)):
            shape = (b, self.partition[i]//self.kernel_size[i], self.kernel_size[i]) + s
            groups[i] = groups[i].view(shape).max(2)[0] 
        return torch.cat(groups,1)

    def extra_repr(self):
        return 'ratio={ratio}, kernel_size={kernel_size}'.format(**self.__dict__)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class VGG(torch.nn.Module):
    
    def __init__(self, n_classes=100):
        
        super(VGG, self).__init__()
        self.chnls = [129, 258, 516, 516, 1032, 1032, 1032, 1032, 1024, 100]
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
        self.mplc=MaxPoolChannels()
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(192, self.chnls[8]),
            torch.nn.BatchNorm1d(self.chnls[8]),
            torch.nn.ELU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(self.chnls[8], self.chnls[9]),
        )

    def forward(self, x):
        lst=[]
        x = self.conv1(x)
        x = self.btn1(x)
        x = self.elu(x)
        y = self.mplc(x)
        lst.append(y)
        x = self.mpl(x)

        x = self.conv2(x)
        x = self.btn2(x)
        x = self.elu(x)
        y = self.mplc(x)
        lst.append(y)
        x = self.mpl(x)

        x = self.conv3(x)
        x = self.btn3(x)
        x = self.elu(x)
        y = self.mplc(x)
        lst.append(y)
        x = self.conv4(x)
        x = self.btn4(x)
        x = self.elu(x)
        y = self.mplc(x)
        lst.append(y)
        x = self.mpl(x)

        x = self.conv5(x)
        x = self.btn5(x)
        x = self.elu(x)
        y = self.mplc(x)
        lst.append(y)
        x = self.conv6(x)
        x = self.btn6(x)
        x = self.elu(x)
        y = self.mplc(x)
        lst.append(y)
        x = self.mpl(x)

        x = self.conv7(x)
        x = self.btn7(x)
        x = self.elu(x)
        y = self.mplc(x)
        lst.append(y)
        x = self.conv8(x)
        x = self.btn8(x)
        x = self.elu(x)
        x = self.mplc(x)
        y=x
        lst.append(y)
        x = self.avgpl(x)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x,lst

class OneCycle(object):
    def __init__(self, nb, max_lr, low_lr, dec_scale):
        self.nb = nb
        self.div = max_lr/low_lr
        self.high_lr = max_lr
        self.iteration = 0
        self.lrs = []
        self.dec_scale = dec_scale
        self.step_len =  int(self.nb / 4)
        
    def calc(self):
        lr = self.calc_lr_cosine()
        self.iteration += 1
        return lr

    def calc_lr_cosine(self):
        if self.iteration ==  0:
            self.lrs.append(self.high_lr/self.div)
            return self.high_lr/self.div
        elif self.iteration == self.nb:
            self.iteration = 0
            self.lrs.append(self.high_lr/self.div)
            old_high_lr = self.high_lr
            old_div = self.div
            self.high_lr = self.high_lr/self.dec_scale
            self.div = self.div/self.dec_scale
            return old_high_lr/old_div
        elif self.iteration > self.step_len:
            ratio = (self.iteration -self.step_len)/(self.nb - self.step_len)
            lr = (self.high_lr/self.div) + 0.5 * (self.high_lr - self.high_lr/self.div) * (1 + math.cos(math.pi * ratio))
        else :
            ratio = self.iteration/self.step_len
            lr = self.high_lr - 0.5 * (self.high_lr - self.high_lr/self.div) * (1 + math.cos(math.pi * ratio))
        self.lrs.append(lr)
        return lr

params = {
    'weight_decay_rate' : 1e-7,
    'cyc_size' : 70,
    'max_lr' : 5e-3,
    'low_lr' : 1e-5,
    'dec_scale' : 1,
    'batch_size' : 32,
    'mode' : 'train',
    'tot_epochs' : 90,
    'beta_weight' : 1,
    'num_layers' : 8,
}

data_transforms = {'train' : transforms.Compose([transforms.ToTensor()]),
                    'val' : transforms.Compose([transforms.ToTensor()])}
data_dir = 'ImageNet-rot-masked/'
image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
                  'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])}
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=params['batch_size'], shuffle=True, num_workers=2),
                'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=params['batch_size'], shuffle=False, num_workers=2)}
dataset_sizes = {'train': len(image_datasets['train']),
                  'val': len(image_datasets['val'])}

model = VGG().to(device)
classification_loss = nn.CrossEntropyLoss()
equivariance_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['low_lr'], weight_decay=params['weight_decay_rate'])
onecycle = OneCycle(math.ceil(dataset_sizes['train']/params['batch_size'])*params['cyc_size'], params['max_lr'], params['low_lr'], params['dec_scale'])
all_rotations = [(RandomRotation((x,x)),RandomRotation((-x,-x))) for x in range(45,360,45)]

best_val_acc = 0
for epoch in range(params['tot_epochs']):
    model.train()
    running_loss1 = 0.0
    running_loss2 = [0.0]*params['num_layers']
    running_corrects = 0
    for i,(inputs, labels) in enumerate(tqdm(dataloaders['train'])):
        inputs = inputs.to(device)
        labels = labels.to(device)
        tbs = labels.shape[0]
        ar = random.choices(all_rotations,k=tbs)
        for i in range(tbs):
            inputs = torch.cat((inputs,(ar[i][0](inputs[i])).unsqueeze(0)),0)
 
        optimizer.zero_grad()
        if epoch<params['cyc_size']:    
            lr = onecycle.calc()
            for g in optimizer.param_groups:
                    g['lr'] = lr
        outputs, temparr = model(inputs)

        loss2 = []
        for j in range(params['num_layers']):
            temp_rot = (ar[0][1](temparr[j][tbs])).unsqueeze(0)
            for i in range(1,tbs):
                temp_rot = torch.cat((temp_rot,(ar[i][1](temparr[j][tbs+i])).unsqueeze(0)),0)
            loss2.append(equivariance_loss(temparr[j][:tbs],temp_rot))
      
        labels = torch.cat((labels,labels))
        _, preds = torch.max(outputs, 1)
        loss1 = classification_loss(outputs, labels)
    
        (loss1+params['beta_weight']*loss2[0]+params['beta_weight']*loss2[1]+params['beta_weight']*loss2[2]+params['beta_weight']*loss2[3]
            +params['beta_weight']*loss2[4]+params['beta_weight']*loss2[5]+params['beta_weight']*loss2[6]+params['beta_weight']*loss2[7]).backward()
        optimizer.step()
        running_loss1 += loss1.item() * inputs.size(0)
        for i in range(params['num_layers']):
            running_loss2[i] += loss2[i].item() * tbs
        running_corrects += torch.sum(preds == labels.data)
        sys.stdout.flush()
          
    train_class_loss = running_loss1 / (dataset_sizes['train']*2)
    train_equi_loss = [running_loss2[i] / (dataset_sizes['train']) for i in range(params['num_layers'])]
    train_acc = running_corrects.double() / (dataset_sizes['train']*2)
    train_tot_loss = train_class_loss + sum(train_equi_loss)
    
    ### Validation
    model.eval()
    with torch.no_grad():
        val_running_loss1 = 0.0
        val_running_loss2 = [0.0]*params['num_layers']
        val_running_corrects = 0
        
        for i,(inputs, labels) in enumerate(tqdm(dataloaders['val'])):
            inputs = inputs.to(device)
            labels = labels.to(device)
            tbs = labels.shape[0]
            ar = random.choices(all_rotations,k=tbs)
            for i in range(tbs):
                inputs = torch.cat((inputs,(ar[i][0](inputs[i])).unsqueeze(0)),0)
            
            outputs, temparr = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss2 = []
            for j in range(params['num_layers']):
                temp_rot = (ar[0][1](temparr[j][tbs])).unsqueeze(0)
                for i in range(1,tbs):
                    temp_rot = torch.cat((temp_rot,(ar[i][1](temparr[j][tbs+i])).unsqueeze(0)),0)
                loss2.append(equivariance_loss(temparr[j][:tbs],temp_rot).item())
            
            _, preds = torch.max(outputs, 1)
            labels = torch.cat((labels,labels))
            loss1 = classification_loss(outputs, labels)
            val_running_loss1 += loss1.item() * inputs.size(0)
            for i in range(params['num_layers']):
                val_running_loss2[i] += (loss2[i] * tbs)
            val_running_corrects += torch.sum(preds == labels.data)
        
        val_class_loss = val_running_loss1 / (dataset_sizes['val']*2)
        val_equi_loss = [val_running_loss2[i] / dataset_sizes['val'] for i in range(params['num_layers'])]
        val_acc = val_running_corrects.double() / (dataset_sizes['val']*2)
        val_tot_loss = val_class_loss + sum(val_equi_loss)
        
        print(f"Epoch {epoch}: Train Acc {train_acc}, Train Total Loss {train_tot_loss}, Class Loss {train_class_loss}, Equi Loss {train_equi_loss} | ; | Val Acc {val_acc}, Val Total Loss {val_tot_loss}, Class Loss {val_class_loss}, Equi Loss {val_equi_loss}")
        torch.save(model.state_dict(),"last_model_weights.ckpt")
        torch.save(optimizer.state_dict(),"last_adam_settings.ckpt")
        if val_acc>best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),"best_model_weights.ckpt")
