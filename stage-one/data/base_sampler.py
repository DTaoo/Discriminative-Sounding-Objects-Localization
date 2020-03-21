import torch
from torch.utils.data.sampler import Sampler

Class BaseSampler(Sampler):
    def __init__(self):
        super(BaseSampler,self).__init__()

    def __len__(self):

    def __iter__(self):
