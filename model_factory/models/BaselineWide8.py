import numpy as np
import torch
from torch import nn

from model_factory.modules.modules_3d import *

class BaselineWide8(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod = nn.Sequential(
	nn.Conv3d(1, 64, kernel_size = (3,3,3), padding = (1,1,1)),
	nn.BatchNorm3d(64),
	nn.ReLU(),

	Conv3dBlock(64), Conv3dBlock(64), 
        Conv3dDownsample(64),
        Conv3dBlock(128), Conv3dBlock(128), 
	Conv3dDownsample(128),
        Conv3dBlock(256), Conv3dBlock(256), 
        ConvTranspose3dBlock(256), ConvTranspose3dBlock(256), 
        ConvTranspose3dUpsample(256),
        ConvTranspose3dBlock(128), ConvTranspose3dBlock(128), 
        ConvTranspose3dUpsample(128),
	ConvTranspose3dBlock(64), ConvTranspose3dBlock(64), 

        nn.Conv3d(64, 1, kernel_size = (1,1,1)),
        nn.Sigmoid(),
    )

  def forward(self, frames_input):
    frames_output = self.mod(frames_input)
    return frames_output