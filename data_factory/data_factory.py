import numpy as np
import torch
import torchvision.transforms as transforms
import gc

class MMDataset(torch.utils.data.Dataset):
  def __init__(self, file_address, size = 64):
    self.file_address = file_address
    self.size = size
    self.default_size = 64
    self.num_frames_total = 20

    self.transforms = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(self.size),
                                          transforms.ToTensor(),])
    self.make_data_np()
    self.make_data_torch()

    self.length = self.data_torch.shape[0]

  def make_data_np(self):
    data_npz = np.load(self.file_address)
    self.data_np = data_npz["input_raw_data"].reshape((-1,self.num_frames_total,1,self.default_size,self.default_size))
    self.data_np = np.transpose(self.data_np, axes = (0,2,1,3,4))

    del data_npz
    gc.collect()
  
  def make_data_torch(self):
    if self.size == 64:
      self.data_torch = torch.tensor(self.data_np).type(torch.cuda.FloatTensor)
    else:
      self.data_torch = []
      for i in range(self.data_np.shape[0]):
        for t in range(self.data_np.shape[2]):
          self.data_torch.append(self.transforms(self.data_np[i][0][t]))

      self.data_torch = torch.cat(self.data_torch)
      self.data_torch = torch.reshape(self.data_torch, (-1,1,self.num_frames_total,self.size,self.size))
      self.data_torch = self.data_torch.type(torch.cuda.FloatTensor)
      
      del self.data_np
      gc.collect()

  def __getitem__(self, i):
    return self.data_torch[i]

  def __len__(self):
    return self.length
