import torch
from torch import nn, Tensor, einsum
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, repeat, reduce

class VoiceBox(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
