import torch
import numpy as np


def torch_to_numpy(tensor):
  return tensor.detach().cpu().numpy()

def split_frames(frames_all, num_frames_input = 10):
  frames_input = frames_all[:,:,:num_frames_input,:,:]
  frames_target = frames_all[:,:,num_frames_input:,:,:]
  return frames_input, frames_target