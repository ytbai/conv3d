import torch
from torch import nn


class Conv2dGRU(nn.Module):
  # num_hidden: number of feature maps of hidden state
  # num_in: number of input feature maps
  def __init__(self, hidden_channels, in_channels):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.in_channels = in_channels

    self.conv_Wz = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size = (3,3), bias = True, padding = 1)
    self.conv_Wr = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size = (3,3), bias = True, padding = 1)
    self.conv_W = nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size = (3,3), bias = True, padding = 1)

    self.conv_Uz = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = (3,3), bias = False, padding = 1)
    self.conv_Ur = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = (3,3), bias = False, padding = 1)
    self.conv_U = nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size = (3,3), bias = False, padding = 1)

  def forward(self, input_curr = None, h_prev = None):
    if h_prev is None:
      batch_size, _, height, width = input_curr.shape
      h_prev = torch.zeros(batch_size, self.hidden_channels, height, width).type(torch.cuda.FloatTensor)
    
    if input_curr is None:
      z_curr = nn.Sigmoid()(self.conv_Uz(h_prev))
      r_curr = nn.Sigmoid()(self.conv_Ur(h_prev))
      h_hat_curr = nn.Tanh()(self.conv_U(r_curr * h_prev))
    else:
      z_curr = nn.Sigmoid()(self.conv_Wz(input_curr) + self.conv_Uz(h_prev))
      r_curr = nn.Sigmoid()(self.conv_Wr(input_curr) + self.conv_Ur(h_prev))
      h_hat_curr = nn.Tanh()(self.conv_W(input_curr) + self.conv_U(r_curr * h_prev))
    
    h_curr = (1-z_curr) * h_prev + z_curr * h_hat_curr

    return h_curr


class ConvTranspose2dGRU(nn.Module):
  # num_hidden: number of feature maps of hidden state
  # num_in: number of input feature maps
  def __init__(self, hidden_channels, in_channels):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.in_channels = in_channels

    self.conv_Wz = nn.ConvTranspose2d(self.in_channels, self.hidden_channels, kernel_size = (3,3), bias = True, padding = 1)
    self.conv_Wr = nn.ConvTranspose2d(self.in_channels, self.hidden_channels, kernel_size = (3,3), bias = True, padding = 1)
    self.conv_W = nn.ConvTranspose2d(self.in_channels, self.hidden_channels, kernel_size = (3,3), bias = True, padding = 1)

    self.conv_Uz = nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, kernel_size = (3,3), bias = False, padding = 1)
    self.conv_Ur = nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, kernel_size = (3,3), bias = False, padding = 1)
    self.conv_U = nn.ConvTranspose2d(self.hidden_channels, self.hidden_channels, kernel_size = (3,3), bias = False, padding = 1)

  def forward(self, input_curr = None, h_prev = None):
    if h_prev is None:
      batch_size, _, height, width = input_curr.shape
      h_prev = torch.zeros(batch_size, self.hidden_channels, height, width).type(torch.cuda.FloatTensor)
    
    if input_curr is None:
      z_curr = nn.Sigmoid()(self.conv_Uz(h_prev))
      r_curr = nn.Sigmoid()(self.conv_Ur(h_prev))
      h_hat_curr = nn.Tanh()(self.conv_U(r_curr * h_prev))
    else:
      z_curr = nn.Sigmoid()(self.conv_Wz(input_curr) + self.conv_Uz(h_prev))
      r_curr = nn.Sigmoid()(self.conv_Wr(input_curr) + self.conv_Ur(h_prev))
      h_hat_curr = nn.Tanh()(self.conv_W(input_curr) + self.conv_U(r_curr * h_prev))
    
    h_curr = (1-z_curr) * h_prev + z_curr * h_hat_curr

    return h_curr





class Conv2dDownsample(nn.Module):
  def __init__(self, input_channels):
    super().__init__()
    self.input_channels = input_channels
    self.output_channels = input_channels * 2

    self.mod = nn.Sequential(
        nn.Conv2d(self.input_channels, self.output_channels, stride = (2, 2), kernel_size = (3,3), padding = (1,1)),
        nn.BatchNorm2d(self.output_channels),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.mod(x)



class Conv2dUpsample(nn.Module):
  def __init__(self, input_channels):
    super().__init__()

    self.input_channels = input_channels
    self.output_channels = input_channels // 2

    self.mod = nn.Sequential(
        nn.ConvTranspose2d(self.input_channels, self.output_channels, stride = (2,2), kernel_size = (3,3), padding = (1,1), output_padding = (1,1)),
        nn.BatchNorm2d(self.output_channels),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.mod(x)


# Block of conv layers
class Conv2dBlock(nn.Module):
  def __init__(self, input_channels, output_channels = None):
    super().__init__()
    self.input_channels = input_channels
    if output_channels is None:
      self.output_channels = self.input_channels
    else:
      self.output_channels = output_channels

    self.mod = nn.Sequential(
        nn.Conv2d(self.input_channels, self.output_channels, kernel_size = (3,3), padding = (1,1)),
        nn.BatchNorm2d(self.output_channels),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.mod(x)


# Block of conv layers
class ConvTranspose2dBlock(nn.Module):
  def __init__(self, input_channels, output_channels = None):
    super().__init__()
    self.input_channels = input_channels
    if output_channels is None:
      self.output_channels = self.input_channels
    else:
      self.output_channels = output_channels


    self.mod = nn.Sequential(
        nn.ConvTranspose2d(self.input_channels, self.output_channels, kernel_size = (3,3), padding = (1,1)),
        nn.BatchNorm2d(self.output_channels),
        nn.ReLU(),
    )

  def forward(self, x):
    return self.mod(x)


class Conv2dGRUEncoder(nn.Module):
  def __init__(self, hidden_channels, encoder_seg, rec_unit):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.encoder_seg = encoder_seg
    self.rec_unit = rec_unit

  def forward(self, frames_input):
    batch_size, _, num_frames_input, _, _ = frames_input.shape

    h_prev = None
    for t in range(num_frames_input):
      frame_curr = frames_input[:,:,t,:,:]
      encoding_curr = self.encoder_seg(frame_curr)

      h_curr = self.rec_unit(encoding_curr, h_prev = h_prev)
      h_prev = h_curr

    h_encoding = h_prev
    return h_encoding

class Conv2dGRUDecoder(nn.Module):
  def __init__(self, hidden_channels, decoder_seg, encoder_seg, rec_unit):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.decoder_seg = decoder_seg
    self.encoder_seg = encoder_seg
    self.rec_unit = rec_unit

  def forward(self, h_encoding, mode = "self", frames_all = None):
    num_frames_input = 10
    num_frames_all = 20
    batch_size, _, height, width = h_encoding.shape
    h_prev = h_encoding
    frames_pred_list = []
    for t in range(num_frames_all - num_frames_input):
      frames_curr = self.decoder_seg(h_prev)
      frames_pred_list.append(frames_curr)

      if mode == "self":
        input_curr = self.encoder_seg(frames_curr)
      elif mode == "GT":
        input_curr = self.encoder_seg(frames_all[:,:,num_frames_input+t,:,:])
      h_curr = self.rec_unit(input_curr, h_prev = h_prev)

      h_prev = h_curr

    frames_pred = torch.stack(frames_pred_list, axis = 2)
    return frames_pred


class Conv2dRNN(nn.Module):
  def __init__(self, hidden_channels):
    super().__init__()

    self.hidden_channels = hidden_channels
    self.encoded_channels = hidden_channels

    self.encoder_rec_unit = Conv2dGRU(hidden_channels = self.hidden_channels, in_channels = self.encoded_channels)
    self.decoder_rec_unit = ConvTranspose2dGRU(hidden_channels = self.hidden_channels, in_channels = self.encoded_channels)
    
    if hidden_channels == 32:
      self.encoder_seg = Conv2dEncoderSmall()
      self.decoder_seg = Conv2dDecoderSmall()
    elif hidden_channels == 64:
      self.encoder_seg = Conv2dEncoderLarge()
      self.decoder_seg = Conv2dDecoderLarge()

    self.encoder_seq = Conv2dGRUEncoder(self.hidden_channels, self.encoder_seg, self.encoder_rec_unit)
    self.decoder_seq = Conv2dGRUDecoder(self.hidden_channels, self.decoder_seg, self.encoder_seg, self.decoder_rec_unit)


  def pred_frames(self, frames_all):
    num_frames_input = 10
    frames_input = frames_all[:,:,num_frames_input,:,:]
    h_encoding = self.encoder_seq(frames_input)
    frames_pred = self.decoder_seq(h_encoding, mode = "GT", frames_all = frames_all)
    return frames_pred

  def forward(self, frames_input):
    h_encoding = self.encoder_seq(frames_input)
    frames_pred = self.decoder_seq(h_encoding)
    return frames_pred


class Conv2dEncoderSmall(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod = nn.Sequential(
	nn.Conv2d(1, 8, kernel_size = (3,3), padding = (1,1)),
	nn.BatchNorm2d(8),
	nn.ReLU(),
	Conv2dBlock(8), Conv2dBlock(8), 
        Conv2dDownsample(8),
        Conv2dBlock(16), Conv2dBlock(16), 
	Conv2dDownsample(16),
        Conv2dBlock(32), Conv2dBlock(32), 
    )

  def forward(self, x):
    return self.mod(x)


class Conv2dDecoderSmall(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod = nn.Sequential(
        ConvTranspose2dBlock(32), ConvTranspose2dBlock(32),
        Conv2dUpsample(32),
        ConvTranspose2dBlock(16), ConvTranspose2dBlock(16),
        Conv2dUpsample(16),
	ConvTranspose2dBlock(8), ConvTranspose2dBlock(8), 
        nn.Conv2d(8, 1, kernel_size = (1,1)),
        nn.Sigmoid(),
    )

  def forward(self, x):
    return self.mod(x)


class Conv2dEncoderLarge(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod = nn.Sequential(
	nn.Conv2d(1, 16, kernel_size = (3,3), padding = (1,1)),
	nn.BatchNorm2d(16),
	nn.ReLU(),
	Conv2dBlock(16), Conv2dBlock(16), 
        Conv2dDownsample(16),
        Conv2dBlock(32), Conv2dBlock(32), 
	Conv2dDownsample(32),
        Conv2dBlock(64), Conv2dBlock(64), 
    )

  def forward(self, x):
    return self.mod(x)


class Conv2dDecoderLarge(nn.Module):
  def __init__(self):
    super().__init__()
    self.mod = nn.Sequential(
        ConvTranspose2dBlock(64), ConvTranspose2dBlock(64),
        Conv2dUpsample(64),
        ConvTranspose2dBlock(32), ConvTranspose2dBlock(32),
        Conv2dUpsample(32),
	ConvTranspose2dBlock(16), ConvTranspose2dBlock(16), 
        nn.Conv2d(16, 1, kernel_size = (1,1)),
        nn.Sigmoid(),
    )

  def forward(self, x):
    return self.mod(x)
