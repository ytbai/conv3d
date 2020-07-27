# Video Prediction With 3d Convolutional Networks

This repository provides code for a project I did on video prediction using 3d convolutional neural networks. The details are presented in a [blog post](https://ytbai.net/2020/07/19/video-prediction-with-3d-convolutional-nets/) I wrote. The code here can be used to reproduce all the results from that post.

## Quick Guide

```main.ipynb``` --- Notebook explaining how to use the code
```plot_maker.ipynb``` --- Notebook for making all the plots in the blog post
```model_factory/models/*``` --- Python scripts for building the neural network models
```model_factory/state_dict/*``` --- Saved weights for the neural network models
```data_factory/data/*``` --- Moving MNIST dataset, copied directly from the [E3D-LSTM repository](https://github.com/google/e3d_lstm) by Yang, et al.