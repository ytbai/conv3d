import numpy as np
import torch
from torch import nn

from model_factory.modules.modules_3d import *

class Baseline(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod = nn.Sequential(
	nn.Conv3d(1, 8, kernel_size = (3,3,3), padding = (1,1,1)),
	nn.BatchNorm3d(8),
	nn.ReLU(),

	Conv3dBlock(8), Conv3dBlock(8), 
        Conv3dDownsample(8),
        Conv3dBlock(16), Conv3dBlock(16), 
	Conv3dDownsample(16),
        Conv3dBlock(32), Conv3dBlock(32), 
        ConvTranspose3dBlock(32), ConvTranspose3dBlock(32), 
        ConvTranspose3dUpsample(32),
        ConvTranspose3dBlock(16), ConvTranspose3dBlock(16), 
        ConvTranspose3dUpsample(16),
	ConvTranspose3dBlock(8), ConvTranspose3dBlock(8), 

        nn.Conv3d(8, 1, kernel_size = (1,1,1)),
        nn.Sigmoid(),
    )

  def forward(self, frames_input):
    frames_output = self.mod(frames_input)
    return frames_output