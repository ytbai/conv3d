import torch
from torch import nn
import numpy as np

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from eval.utils import *




def evaluate_mse(model, test_dataset, batch_size = 32):
  model.eval()
  criterion = nn.MSELoss()
  loss_total = 0
  num_samples = 0
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
  for frames_all in test_dataloader:
    frames_input, frames_target = split_frames(frames_all)
    frames_pred = model(frames_input)
    loss_batch_total = criterion(frames_pred, frames_target).item()
    loss_batch_total *= frames_all.shape[0]
    loss_batch_total *= frames_input.shape[3] * frames_input.shape[4]
    loss_total += loss_batch_total
    num_samples += frames_all.shape[0]


  #print(loss_total, num_samples)
  return loss_total / num_samples


def psnr_single(image_1, image_2, dynamic_range):
  image_1 = torch_to_numpy(image_1)
  image_2 = torch_to_numpy(image_2)  
  
  return peak_signal_noise_ratio(image_1, image_2, data_range = dynamic_range)

def psnr_multiple(images_1, images_2, dynamic_range):
  num_frames = images_1.shape[0]
  psnr_list = []
  for t in range(num_frames):
    curr_psnr = psnr_single(images_1[t], images_2[t], dynamic_range)
    psnr_list.append(curr_psnr)
  return np.array(psnr_list)

def ssim_single(image_1, image_2, range_min, range_max):
  image_1 = torch_to_numpy(image_1)-range_min
  image_2 = torch_to_numpy(image_2)-range_min
  dynamic_range = range_max - range_min

  return structural_similarity(image_1, image_2, data_range = dynamic_range, gaussian_weights = True, use_sample_covariance = False, sigma = 1.5)
  
def ssim_multiple(images_1, images_2, range_min, range_max):
  num_frames = images_1.shape[0]
  ssim_list = []
  for t in range(num_frames):
    curr_ssim = ssim_single(images_1[t], images_2[t], range_min, range_max)
    ssim_list.append(curr_ssim)

  return np.array(ssim_list)


def evaluate_psnr(model, test_dataset):
  num_frames_input = 10
  num_frames_all = 20
  num_frames_output = num_frames_all - num_frames_input
  psnr_mean = np.zeros(num_frames_output)
  num_samples = 0

  model.eval()
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
  for frames_all in test_dataloader:
    frames_input, frames_target = split_frames(frames_all, num_frames_input)
    frames_pred = model(frames_input)
    size = frames_target.shape[-1]

    psnr_mean += psnr_multiple(
        frames_pred.reshape(num_frames_output,size,size), 
        frames_target.reshape(num_frames_output,size,size), 
        dynamic_range = 1.)
    num_samples += 1

  psnr_mean = psnr_mean/num_samples
  #plt.plot(range(1,11), psnr_mean)
  return psnr_mean


def evaluate_ssim(model, test_dataset):
  range_min = 0
  range_max = 1
  num_frames_input = 10
  num_frames_all = 20
  num_frames_output = num_frames_all - num_frames_input
  ssim_mean = np.zeros(num_frames_output)
  num_samples = 0

  model.eval()
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
  for frames_all in test_dataloader:
    frames_input, frames_target = split_frames(frames_all, num_frames_input)
    frames_pred = model(frames_input)
    size = frames_target.shape[-1]

    ssim_mean += ssim_multiple(
        frames_pred.reshape(num_frames_output,size,size), 
        frames_target.reshape(num_frames_output,size,size), 
        range_min = range_min, 
        range_max = range_max)
    num_samples += 1

  ssim_mean = ssim_mean/num_samples
  #plt.plot(range(1,11), ssim_mean)
  return ssim_mean