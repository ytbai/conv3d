import torch
import numpy as np
from eval.utils import *
import matplotlib.pyplot as plt

def show_frames(frames, ax, width_per_frame, row_label = None):
  frames = torch_to_numpy(frames)
  _, channels, num_frames, _, _ = frames.shape
  
  for t in range(num_frames):
    ax[t].imshow(frames[0][0][t])
    ax[t].set_xticks([])
    ax[t].set_yticks([])
  if row_label is not None:
    ax[0].set_ylabel(row_label)

def show_result(model, dataset, index):
  num_frames_input = 10
  width_per_frame = 5
  
  model.eval()
  frames_all = dataset[index].unsqueeze(0)
  frames_input, frames_target = split_frames(frames_all, num_frames_input)
  frames_pred = model(frames_input)

  fig, ax = plt.subplots(3, num_frames_input, figsize = (5*10/3, 5))

  show_frames(frames_input, ax[0], width_per_frame, row_label = "Context")
  show_frames(frames_target, ax[1], width_per_frame, row_label = "Target")
  show_frames(frames_pred, ax[2], width_per_frame, row_label = "Prediction")