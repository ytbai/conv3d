import numpy as np
import torch
from torch import nn

from model_factory.modules.modules_3d import *


class BaselineWide4Deep4(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod = nn.Sequential(
	nn.Conv3d(1, 32, kernel_size = (3,3,3), padding = (1,1,1)),
	nn.BatchNorm3d(32),
	nn.ReLU(),

	Conv3dBlock(32), Conv3dBlock(32), Conv3dBlock(32), Conv3dBlock(32), 
        Conv3dDownsample(32),
        Conv3dBlock(64), Conv3dBlock(64), Conv3dBlock(64), Conv3dBlock(64), 
	Conv3dDownsample(64),
        Conv3dBlock(128), Conv3dBlock(128), Conv3dBlock(128), Conv3dBlock(128), 
        ConvTranspose3dBlock(128), ConvTranspose3dBlock(128), ConvTranspose3dBlock(128), ConvTranspose3dBlock(128), 
        ConvTranspose3dUpsample(128),
        ConvTranspose3dBlock(64), ConvTranspose3dBlock(64), ConvTranspose3dBlock(64), ConvTranspose3dBlock(64), 
        ConvTranspose3dUpsample(64),
	ConvTranspose3dBlock(32), ConvTranspose3dBlock(32), ConvTranspose3dBlock(32), ConvTranspose3dBlock(32), 

        nn.Conv3d(32, 1, kernel_size = (1,1,1)),
        nn.Sigmoid(),
    )

  def forward(self, frames_input):
    frames_output = self.mod(frames_input)
    return frames_output