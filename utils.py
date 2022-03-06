import torch
from torch.utils.data import Dataset
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomRotation, Pad, Resize, ToTensor, Compose
import torchvision.datasets as datasets
import math, os
from torch.nn import Conv2d
import numpy as np
from PIL import Image

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

def DatasetLoader(dataset, batch_size, train=True, resize=None):
    totensor = transforms.ToTensor()
    return_dict = {}
    if dataset=="rot_mnist":
        train_transform = Compose([totensor,])
        test_transform = Compose([totensor,])
        if train:
            mnist_train = MnistRotDataset(mode='train', transform=train_transform)
            return_dict['train_loader'] = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size)
            return_dict['train_size'] = 12000
        mnist_test = MnistRotDataset(mode='test', transform=test_transform)
        return_dict['test_loader']  = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size)
        return_dict['test_size'] = 50000
    else:
        data_dir = dataset
        if resize:
            resize1 = transforms.Resize(resize)
            train_transforms = transforms.Compose([resize1,totensor])
            test_transforms = transforms.Compose([resize1,totensor])
        else:
            train_transforms = transforms.Compose([totensor])
            test_transforms = transforms.Compose([totensor])
        if train:
            train_images = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
            return_dict['train_loader'] = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=2)
            return_dict['train_size'] = len(train_images)
        test_images = datasets.ImageFolder(os.path.join(data_dir, 'val'), test_transforms)
        return_dict['test_loader'] = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=False, num_workers=2)
        return_dict['test_size'] = len(test_images)
    return return_dict

def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
