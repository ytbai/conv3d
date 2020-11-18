import numpy as np
import torch
from torch import nn

from model_factory.modules.modules_3d import *

class BaselineAutoreg(nn.Module):
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

        nn.Conv3d(8, 1, kernel_size = (10,1,1)),
        nn.Sigmoid(),
    )

  def pred_frame(self, x):
    return self.mod(x)


  def pred_frames(self, frames_all):
    num_frames_input = 10
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


