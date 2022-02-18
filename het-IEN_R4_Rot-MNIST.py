import torch
from torch.utils.data import Dataset
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, ELU
from torchvision.transforms import RandomRotation, Pad, Resize, ToTensor, Compose
from torchvision import transforms
import torch.nn.functional as F

import random, time, math, os, pickle, sys
import numpy as np
from PIL import Image
from typing import List, Tuple, Any, Union
from itertools import accumulate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class MaxPoolChannels(torch.nn.Module):   
    def __init__(self):
        super(MaxPoolChannels, self).__init__()
        self.ratio = [8,2,1]
        self.kernel_size = [4,2,1]
        self.sum_ratio = sum(self.ratio)
        
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

class CNNBasic(torch.nn.Module): 

    def __init__(self, n_classes=10):
        super(CNNBasic, self).__init__()
        self.conv1 = Conv2d(1, 88, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1), bias=False)
        self.btn1 = BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = Conv2d(88, 132, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.btn2 = BatchNorm2d(132, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = Conv2d(132, 176, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.btn3 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = Conv2d(176,176, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.btn4 = BatchNorm2d(176, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = Conv2d(176,264, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.btn5 = BatchNorm2d(264, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = Conv2d(264, 352, kernel_size=(5, 5), stride=(1, 1), bias=False)
        self.btn6 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.mpl = MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=(1, 1), ceil_mode=False)
        self.avgpl = AdaptiveAvgPool2d(1)
        self.elu = ELU(alpha=1.0,inplace=True)
        self.mplc=MaxPoolChannels()
        self.fullnet = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ELU(alpha=1,inplace=True),
            torch.nn.Linear(128, n_classes),
        )

    def forward(self, x):
        lst=[]
        
        x = self.conv1(x)
        x = self.btn1(x)
        x = self.elu(x)
        y=  self.mplc(x)
        lst.append(y)
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
        x = self.mplc(x)
        y = x
        lst.append(y)
        x = self.avgpl(x)

        x = self.fullnet(x.reshape(x.shape[0], -1))
        
        return x,lst


class MnistRotDataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']
            
        if mode == "train":
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat"
        else:
            file = "mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat"
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)

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

def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

params = {
    'weight_decay_rate' : 1e-7,
    'cyc_size' : 70,
    'max_lr' : 5e-3,
    'low_lr' : 1e-5,
    'dec_scale' : 1,
    'batch_size' : 64,
    'train_dataset_size' : 12000,
    'val_dataset_size' : 50000,
    'mode' : 'train',
    'tot_epochs' : 90,
    'beta_weight' : 1,
    'num_layers' : 6,
}

model = CNNBasic().to(device)
model.apply(init_weights)
classification_loss = torch.nn.CrossEntropyLoss()
equivariance_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params['low_lr'], weight_decay=params['weight_decay_rate'])
onecycle = OneCycle(math.ceil(params['train_dataset_size']/params['batch_size'])*params['cyc_size'], params['max_lr'], params['low_lr'], params['dec_scale'])

totensor = ToTensor()
train_transform = Compose([totensor,])
mnist_train = MnistRotDataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=params['batch_size'])
test_transform = Compose([totensor,])
mnist_test = MnistRotDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=params['batch_size'])


best_val_acc = 0
for epoch in range(params['tot_epochs']):
    model.train()
    running_loss1 = 0.0
    running_loss2 = [0.0]*params['num_layers']
    running_corrects = 0
    for _, (x, y_t) in enumerate(train_loader):        
        x = x.to(device)
        batch_size=y_t.shape[0]
        ar = random.choices(range(1,4),k=batch_size)
        for i in range(batch_size):
            x = torch.cat((x,torch.rot90(x[i], ar[i], [-2, -1]).unsqueeze(0)),0)

        optimizer.zero_grad()
        if epoch<params['cyc_size']:    
            lr = onecycle.calc()
            for g in optimizer.param_groups:
                    g['lr'] = lr

        y_t = y_t.to(device)
        y_t = torch.cat((y_t,y_t))
        y_h, temparr = model(x)
        loss1 = classification_loss(y_h, y_t)

        _, preds = torch.max(y_h.data, 1)

        loss2=[]
        for j in range(params['num_layers']):
            temp_rot = torch.rot90(temparr[j][batch_size], -ar[0], [-2,-1]).unsqueeze(0)
            for i in range(1,batch_size):
                temp_rot = torch.cat((temp_rot,torch.rot90(temparr[j][batch_size+i],-ar[i],[-2,-1]).unsqueeze(0)),0)
            loss2.append(equivariance_loss(temparr[j][:batch_size],temp_rot))
        
        (loss1+params['beta_weight']*loss2[0]+params['beta_weight']*loss2[1]+params['beta_weight']*loss2[2]+
            params['beta_weight']*loss2[3]+params['beta_weight']*loss2[4]+params['beta_weight']*loss2[5]).backward()
        
        optimizer.step()
        running_loss1 += loss1.item() * x.size(0)
        for i in range(params['num_layers']):
            running_loss2[i] += loss2[i].item() * batch_size
        running_corrects += torch.sum(preds == y_t.data)
        sys.stdout.flush()

    train_class_loss = running_loss1 / (params['train_dataset_size']*2)
    train_equi_loss = [running_loss2[i] / params['train_dataset_size'] for i in range(params['num_layers'])]
    train_acc = running_corrects.double() / (params['train_dataset_size']*2)
    train_tot_loss = train_class_loss + sum(train_equi_loss)

    model.eval()
    with torch.no_grad():
        val_running_loss1 = 0.0
        val_running_loss2 = [0.0]*params['num_layers']
        val_running_corrects = 0
      
        for i,(inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            tbs = labels.shape[0]
            ar = random.choices(range(1,4),k=tbs)
            for i in range(tbs):
                inputs = torch.cat((inputs,torch.rot90(inputs[i], ar[i], [-2, -1]).unsqueeze(0)),0)

            outputs, temparr = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            loss2 = []
            for j in range(params['num_layers']):
                temp_rot = torch.rot90(temparr[j][tbs], -ar[0], [-2,-1]).unsqueeze(0)
                for i in range(1,tbs):
                    temp_rot = torch.cat((temp_rot,torch.rot90(temparr[j][tbs+i],-ar[i],[-2,-1]).unsqueeze(0)),0)
                loss2.append(equivariance_loss(temparr[j][:tbs],temp_rot).item())

            _, preds = torch.max(outputs, 1)
            labels = torch.cat((labels,labels))
            loss1 = classification_loss(outputs, labels)
            val_running_loss1 += loss1.item() * inputs.size(0)
            for i in range(params['num_layers']):
                val_running_loss2[i] += (loss2[i] * tbs)
            val_running_corrects += torch.sum(preds == labels.data)
        
        val_class_loss = val_running_loss1 / (params['val_dataset_size']*2)
        val_equi_loss = [val_running_loss2[i] / params['val_dataset_size'] for i in range(params['num_layers'])]
        val_acc = val_running_corrects.double() / (params['val_dataset_size']*2)
        val_tot_loss = val_class_loss + sum(val_equi_loss)

        print(f"Epoch {epoch}: Train Acc {train_acc}, Val Acc {val_acc} ; Train Total Loss {train_tot_loss}, Class Loss {train_class_loss}, Equi Loss {train_equi_loss} | ; | Val Total Loss {val_tot_loss}, Class Loss {val_class_loss}, Equi Loss {val_equi_loss}")
        
        torch.save(model.state_dict(),"last_model_weights.ckpt")
        torch.save(optimizer.state_dict(),"last_adam_settings.ckpt")
        if val_acc>best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),"best_model_weights.ckpt")
