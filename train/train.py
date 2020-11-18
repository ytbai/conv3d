import torch
from eval import *


def train_once(model, criterion, num_frames_input, train_dataloader, optimizer, mode):
  model.train()
  loss_train_epoch = []

  for frames_all in train_dataloader:
    frames_input, frames_target = split_frames(frames_all, num_frames_input)

    if mode == "gt":
      frames_pred = model.pred_frames(frames_all)
    elif mode == "reg":
      frames_pred = model(frames_input)

    optimizer.zero_grad()
    loss_train = criterion(frames_pred, frames_target)
    loss_train.backward()
    optimizer.step()
    loss_train_epoch.append(loss_train.item())

  loss_train_epoch = np.array(loss_train_epoch).mean()
  return loss_train_epoch



def valid_once(model_meta, criterion, num_frames_input, val_dataloader):
  model.eval()
  loss_valid_epoch = []
  for frames_all in val_dataloader:
    frames_input, frames_target = split_frames(frames_all, num_frames_input)
    
    frames_pred = model(frames_input)

    loss_valid = criterion(frames_pred, frames_target)
    
    loss_valid_epoch.append(loss_valid.item())
  loss_valid_epoch = np.array(loss_valid_epoch).mean()
  return loss_valid_epoch
