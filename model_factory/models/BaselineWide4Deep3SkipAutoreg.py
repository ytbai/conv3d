import numpy as np
import torch
from torch import nn

from model_factory.modules.modules_3d import *

class BaselineWide4Deep3SkipAutoreg(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod_1 = nn.Sequential(
	nn.Conv3d(1, 32, kernel_size = (3,3,3), padding = (1,1,1)),
	nn.BatchNorm3d(32),
	nn.ReLU(),
	Conv3dBlock(32), Conv3dBlock(32), Conv3dBlock(32),
        Conv3dDownsample(32),
        Conv3dBlock(64), Conv3dBlock(64), Conv3dBlock(64),
    )

    self.mod_2 = nn.Sequential(
	Conv3dDownsample(64),
        Conv3dBlock(128), Conv3dBlock(128), Conv3dBlock(128),
        ConvTranspose3dBlock(128), ConvTranspose3dBlock(128), ConvTranspose3dBlock(128),
        ConvTranspose3dUpsample(128),
    )

    self.mod_1_2 = nn.Sequential(
	ConvTranspose3dBlock(64*2, 64),
    )

    self.mod_3 = nn.Sequential(
        ConvTranspose3dBlock(64), ConvTranspose3dBlock(64), ConvTranspose3dBlock(64),
        ConvTranspose3dUpsample(64),
	ConvTranspose3dBlock(32), ConvTranspose3dBlock(32), ConvTranspose3dBlock(32),
        nn.Conv3d(32, 1, kernel_size = (10,1,1)),
        nn.Sigmoid(),
    )

  def pred_frame(self, x):
    mod_1_x = self.mod_1(x)
    mod_2_x = self.mod_2(mod_1_x)
    mod_1_2_x = self.mod_1_2(torch.cat([mod_1_x, mod_2_x], dim = 1))
    mod_3_x = self.mod_3(mod_1_2_x)
    return mod_3_x


  def pred_frames(self, frames_all):
    nun_frames_input = 10
    num_frames_all = 20
    frames_pred = []
    for t in range(num_frames_all-num_frames_input):
      frames_input = frames_all[:,:,t:t+num_frames_input,:,:]
      frame_pred = self.pred_frame(frames_input)
      frames_pred.append(frame_pred)

    return torch.cat(frames_pred, dim=2)
      
  def forward(self, frames_input):
    frames_input = frames_input.clone()
    num_frames_input = 10
    for t in range(num_frames_input):
      new_frame = self.pred_frame(frames_input[:,:,t:t+num_frames_input,:,:])
      frames_input = torch.cat([frames_input, new_frame], dim = 2)
    return frames_input[:,:,num_frames_input:,:,:]


