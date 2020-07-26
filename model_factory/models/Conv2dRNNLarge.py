import numpy as np
import torch
from torch import nn

from model_factory.modules.modules_2d import *


class Conv2dRNNLarge(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod = Conv2dRNN(hidden_channels = 64)

  def pred_frames(self, frames_all):
    return self.mod.pred_frames(frames_all)

  def forward(self, x):
    return self.mod(x)