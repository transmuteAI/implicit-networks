import torch
from typing import List
from itertools import accumulate

class HomogeneousChannelsMaxPool(torch.nn.Module):
    def __init__(self, kernel_size: int):
        super(HomogeneousChannelsMaxPool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.shape[1] % self.kernel_size == 0, '''
            Error! The input number of channels ({}) is not divisible by the max pooling kernel size ({})
        '''.format(input.shape[1], self.kernel_size)
        b = input.shape[0]
        c = input.shape[1] // self.kernel_size
        s = input.shape[2:]
        shape = (b, c, self.kernel_size) + s
        return input.view(shape).max(2)[0]

    def extra_repr(self):
        return 'kernel_size={kernel_size}'.format(**self.__dict__)


class HeterogeneousChannelsMaxPool(torch.nn.Module):
    def __init__(self, ratio: List, kernel_size: List):
        super(HeterogeneousChannelsMaxPool, self).__init__()
        self.ratio = ratio
        self.kernel_size = kernel_size
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
