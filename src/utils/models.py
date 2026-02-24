from torch import nn, Tensor
import torch
from typing import Tuple
import torch.nn.functional as F
import numpy as np



class CNN_earlycasevar(nn.Module):

  def __init__(self, input_size_case=1, input_size_process=1, length=5,
               nr_cnn_layers=2, nr_out_channels=10, kernel_size=2, stride=1,
               nr_dense_layers=1, dense_width=20, p=0.1, nr_outputs=1):
    super(CNN, self).__init__()
    self.input_size_case = input_size_case
    self.input_size_process = input_size_process
    self.length = length
    self.nr_cnn_layers = nr_cnn_layers
    self.nr_out_channels = nr_out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.nr_dense_layers = nr_dense_layers
    self.p = p
    self.dense_width = dense_width
    self.nr_outputs = nr_outputs

    def conv1d_size_out(size, kernel_size=self.kernel_size, stride=self.stride):
      return (size - 1 * (kernel_size - 1) - 1) // stride + 1

    # cnn layers
    self.cnn_layers = nn.ModuleDict()
    for nr in range(self.nr_cnn_layers):
      if nr == 0:
        self.cnn_layers[str(nr)] = nn.Sequential(nn.Conv1d(in_channels=self.input_size_process + 1 + self.input_size_case, out_channels=self.nr_out_channels,
                         kernel_size=self.kernel_size, stride=self.stride, dilation=1), nn.BatchNorm1d(self.nr_out_channels))
        conv_size = conv1d_size_out(self.length)
      else:
        self.cnn_layers[str(nr)] = nn.Sequential(nn.Conv1d(in_channels=self.nr_out_channels, out_channels=self.nr_out_channels,
                         kernel_size=self.kernel_size, stride=self.stride, dilation=1), nn.BatchNorm1d(self.nr_out_channels))
        conv_size = conv1d_size_out(conv_size)

    assert conv_size > 0, "too many convolutional layers, too large kernel sizes or strides"

    # compute size of flattened vector
    linear_input_size = conv_size * nr_out_channels

    # dense layers
    self.dense_layers = nn.ModuleDict()
    for nr in range(self.nr_dense_layers):
      if nr == 0:
        self.dense_layers[str(nr)] = nn.Linear(linear_input_size, self.dense_width)
        #self.batchnorm1 = nn.BatchNorm1d(width, affine=False)
      else:
        self.dense_layers[str(nr)] = nn.Linear(self.dense_width, self.dense_width)

    # outputs
    self.last_layer = nn.Linear(self.dense_width, 10)
    self.output_mean = nn.Linear(10, self.nr_outputs)
    self.output_logvar = nn.Linear(10, 1)

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.p)

  def forward(self, x_case, x_process, t=None):
    '''Forward pass'''

    t_reshaped = t.reshape((x_process.shape[0], 1, x_process.shape[2]))
    x = torch.cat((x_process, t_reshaped), 1)

    x_case_reshaped = x_case.repeat(1, 1,  x_process.shape[2]).reshape((x_process.shape[0], 1, x_process.shape[2]))
    x = torch.cat((x, x_case_reshaped), 1)

    x = nn.Sequential(self.cnn_layers[str(0)], self.relu)(x)

    for nr in range(1, self.nr_cnn_layers):
      x = nn.Sequential(self.cnn_layers[str(nr)], self.relu)(x)
      #outputs = self.dropout(outputs)

    x = x.view(x.size(0), -1)

    #x_concat = torch.cat((x_case, x), 1)

    # calculate dense layers
    if self.nr_dense_layers > 0:
      hidden = nn.Sequential(self.dropout, self.dense_layers[str(0)], self.relu)(x)
      for nr in range(1, self.nr_dense_layers):
        hidden = nn.Sequential(self.dropout, self.dense_layers[str(nr)], self.relu)(hidden)
        # x = nn.Sequential(self.dropout, self.layer1, self.relu)(x) #, self.batchnorm1)(x

    # calculate outputs
    last = nn.Sequential(self.dropout, self.last_layer, self.relu)(hidden)
    mean = nn.Sequential(self.dropout, self.output_mean)(last)
    logvar = nn.Sequential(self.dropout, self.output_logvar)(last)

    return mean, logvar


class CNN(nn.Module):

  def __init__(self, input_size_case=1, input_size_process=1, length=5,
               nr_cnn_layers=2, nr_out_channels=10, kernel_size=2, stride=1,
               nr_dense_layers=1, dense_width=20, p=0.1, nr_outputs=1):
    super(CNN, self).__init__()
    self.input_size_case = input_size_case
    self.input_size_process = input_size_process
    self.length = length
    self.nr_cnn_layers = nr_cnn_layers
    self.nr_out_channels = nr_out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.nr_dense_layers = nr_dense_layers
    self.p = p
    self.dense_width = dense_width
    self.nr_outputs = nr_outputs

    def conv1d_size_out(size, kernel_size=self.kernel_size, stride=self.stride):
      return (size - 1 * (kernel_size - 1) - 1) // stride + 1

    # cnn layers
    self.cnn_layers = nn.ModuleDict()
    for nr in range(self.nr_cnn_layers):
      if nr == 0:
        self.cnn_layers[str(nr)] = nn.Sequential(nn.Conv1d(in_channels=self.input_size_process + 1, out_channels=self.nr_out_channels,
                         kernel_size=self.kernel_size, stride=self.stride, dilation=1), nn.BatchNorm1d(self.nr_out_channels))
        conv_size = conv1d_size_out(self.length)
      else:
        self.cnn_layers[str(nr)] = nn.Sequential(nn.Conv1d(in_channels=self.nr_out_channels, out_channels=self.nr_out_channels,
                         kernel_size=self.kernel_size, stride=self.stride, dilation=1), nn.BatchNorm1d(self.nr_out_channels))
        conv_size = conv1d_size_out(conv_size)

    assert conv_size > 0, "too many convolutional layers, too large kernel sizes or strides"

    # compute size of flattened vector
    linear_input_size = conv_size * nr_out_channels

    # dense layers
    self.dense_layers = nn.ModuleDict()
    for nr in range(self.nr_dense_layers):
      if nr == 0:
        self.dense_layers[str(nr)] = nn.Linear(self.input_size_case + linear_input_size, self.dense_width)
        #self.batchnorm1 = nn.BatchNorm1d(width, affine=False)
      else:
        self.dense_layers[str(nr)] = nn.Linear(self.dense_width, self.dense_width)

    # outputs
    self.last_layer = nn.Linear(self.dense_width, 10)
    self.output_mean = nn.Linear(10, self.nr_outputs)
    self.output_logvar = nn.Linear(10, 1)

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.p)

  def forward(self, x_case, x_process, t=None):
    '''Forward pass'''

    t_reshaped = t.reshape((x_process.shape[0], 1, x_process.shape[2]))
    x = torch.cat((x_process, t_reshaped), 1)

    x = nn.Sequential(self.cnn_layers[str(0)], self.relu)(x)

    for nr in range(1, self.nr_cnn_layers):
      x = nn.Sequential(self.cnn_layers[str(nr)], self.relu)(x)
      #outputs = self.dropout(outputs)

    x = x.view(x.size(0), -1)

    x_concat = torch.cat((x_case, x), 1)

    # calculate dense layers
    if self.nr_dense_layers > 0:
      hidden = nn.Sequential(self.dropout, self.dense_layers[str(0)], self.relu)(x_concat)
      for nr in range(1, self.nr_dense_layers):
        hidden = nn.Sequential(self.dropout, self.dense_layers[str(nr)], self.relu)(hidden)
        # x = nn.Sequential(self.dropout, self.layer1, self.relu)(x) #, self.batchnorm1)(x

    # calculate outputs
    last = nn.Sequential(self.dropout, self.last_layer, self.relu)(hidden)
    mean = nn.Sequential(self.dropout, self.output_mean)(last)
    logvar = nn.Sequential(self.dropout, self.output_logvar)(last)

    return mean, logvar

###################################################################################################################################
###################################################################################################################################


class LSTM(nn.Module):
  '''
    Multilayer Perceptron.
  '''

  def __init__(self, input_size_case=1, input_size_process=1,
               nr_lstm_layers=1, lstm_size=1,
               nr_dense_layers=1, dense_width=20, p=0.1, nr_outputs=1, treatment_length=1, iteration=0):
    super().__init__()
    self.input_size_case = input_size_case
    self.input_size_process = input_size_process
    self.nr_lstm_layers = nr_lstm_layers
    self.lstm_size = lstm_size
    self.nr_dense_layers = nr_dense_layers
    self.dense_width = dense_width
    self.p = p         # dropout probability
    self.nr_outputs = nr_outputs
    self.treatment_length = treatment_length
    self.iteration = iteration

    # lstm layers
    if self.nr_lstm_layers > 0:
      self.lstm_layers = nn.ModuleDict()

      for nr in range(self.nr_lstm_layers):
        if nr == 0:
          # self.lstm_layers[str(nr)] = nn.LSTM(self.input_size_process + 1, self.lstm_size, 1)
          self.lstm_layers[str(nr)] = nn.LSTM(self.input_size_process + self.treatment_length, self.lstm_size, 1)
        else:
          self.lstm_layers[str(nr)] = nn.LSTM(self.lstm_size, self.lstm_size, 1)
    else:
      self.lstm_size = 0


    # dense layers
    self.dense_layers = nn.ModuleDict()
    for nr in range(self.nr_dense_layers):
      if nr == 0:
        self.dense_layers[str(nr)] = nn.Linear(self.input_size_case + self.lstm_size, self.dense_width)
        #self.batchnorm1 = nn.BatchNorm1d(width, affine=False)
      else:
        self.dense_layers[str(nr)] = nn.Linear(self.dense_width, self.dense_width)

    # outputs
    self.last_layer = nn.Linear(self.dense_width, 10)
    self.output_mean = nn.Linear(10, self.nr_outputs)
    # self.output_logvar = nn.Linear(10, 1)

    # parameter-free layers
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(self.p)


  def forward(self, x_case, x_process, t=None):
    '''Forward pass'''

    #self.lstm.flatten_parameters()
    # TODO
    # check whether prefixes are correct

    # print(t.shape, "t.shape")
    # print(t, "t")
    # print(x_process.shape, "x_process.shape")
    # print(x_process, "x_process")

    
    t_reshaped = t.reshape((x_process.shape[0], self.treatment_length, x_process.shape[2]))
    # print(t_reshaped.shape)
    # print(t_reshaped)
    x = torch.cat((x_process, t_reshaped), 1)
    # print(x.shape, "x.shape")

    # lstm layers
    if self.nr_lstm_layers > 0:
      x = x.transpose(0, 2)
      x = x.transpose(1, 2)

      outputs, (h, c) = self.lstm_layers[str(0)](x)
      ### !!! ### this dropout is only for standard LSTMs
      outputs = self.dropout(outputs)

      for nr in range(1, self.nr_lstm_layers):
        outputs, (h, c) = self.lstm_layers[str(nr)](outputs, (h, c))
        outputs = self.dropout(outputs)

      outputs = outputs[-1, :, :]
      # concatenate lstm output with case variables
      x_concat = torch.cat((x_case, outputs), 1)
    else:
      x_concat = x_case

      x_concat = self.relu(x_concat)

    # calculate dense layers
    if self.nr_dense_layers > 0:
      hidden = nn.Sequential(self.dropout, self.dense_layers[str(0)], self.relu)(x_concat)
      for nr in range(1, self.nr_dense_layers):
        hidden = nn.Sequential(self.dropout, self.dense_layers[str(nr)], self.relu)(hidden)
        # x = nn.Sequential(self.dropout, self.layer1, self.relu)(x) #, self.batchnorm1)(x

    # calculate outputs
    last = nn.Sequential(self.dropout, self.last_layer, self.relu)(hidden)
    mean = nn.Sequential(self.dropout, self.output_mean)(last)
    # logvar = nn.Sequential(self.dropout, self.output_logvar)(last)

    return mean
    # return mean, logvar

