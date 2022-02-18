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
    def __init__(self, num_classes=100, dropout_rate=0, deltaorth=False, initial_stride=1, fixparams=True):
        super(ResNet18, self).__init__()
        nStages = [129, 129, 258, 516, 1032]

        self.conv1 = conv3x3(3,nStages[0])
        self.in_planes = nStages[0]
        self.layer1 = self._wide_layer(wide_basic, nStages[1], 2, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], 2, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], 2, dropout_rate, stride=2)
        self.layer4 = self._wide_layer(wide_basic, nStages[4], 2, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[4], momentum=0.9)
        self.avgpool = AdaptiveAvgPool2d((1,1))
        self.gpool = MaxPoolChannels()
        self.linear = nn.Linear(192, num_classes)

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
        out1 = self.gpool(out)
        out = self.layer2(out)
        out2 = self.gpool(out)
        out = self.layer3(out)
        out3 = self.gpool(out)
        out = self.layer4(out)
        out = F.relu(self.bn1(out))
        out = self.gpool(out)
        out4=out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out,[out1,out2,out3,out4]

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
    'num_layers' : 4,
}

resize1 = Resize(65)
data_transforms = {'train' : transforms.Compose([resize1,transforms.ToTensor()]),
                    'val' : transforms.Compose([resize1,transforms.ToTensor()])}
data_dir = 'ImageNet-rot-masked/'
image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
                  'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])}
dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=params['batch_size'], shuffle=True, num_workers=2),
                'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=params['batch_size'], shuffle=False, num_workers=2)}
dataset_sizes = {'train': len(image_datasets['train']),
                  'val': len(image_datasets['val'])}

model = ResNet18().to(device)
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
    
        (loss1+params['beta_weight']*loss2[0]+params['beta_weight']*loss2[1]+params['beta_weight']*loss2[2]+params['beta_weight']*loss2[3]).backward()
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
