# Neural networks comprise of layers / modules that perform operations on data. The torch.nn namespace provides
# the building blocks you need to build your own neural network. Every module in PyTorch subclasses the nn.Module.
# A neural network is a module itself that consists of other modules (layers). This nested structure allows for
# building and managing complex architectures easily.

# In the following sections, we'll build a neural network to classify images in the FashionMNIST dataset.

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Get Device for Training
# We want to be able to train our model on an accelerator such as CUDA, MPS, MTIA, or XPU. If the current accelerator
# is available, we will use it. Otherwise, we use the CPU.

device = torch.accelerator.current_accelerator().type
if torch.accelerator.is_available()
else "cpu"
print(f" Using {device} device")

# Define the Class
# We define our neural netwwork by subclassing nn.Module, and initialize the neural network layers in __init__
# Every nn.module subclass implements the operations on input data in the forward method.

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.Linear(28 * 28 * 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10)
    )
  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear_relu_stack(x)
    return logits
# Here we create an instance of NeuralNetwork, and move it to the device, and print its structure.
model = NeuralNetwork().to(device)
print (model)


# To use the model, we pass it inout data. this executes the model's forward, along with some background operations
# Note: Don't call model.forward() directly.

# Calling the model on the inout returns a 2-dimensional tensor with dim = 0 corresponding to each output of 10 raw
# predicted values for each class, and dim = 1 corresponding to the individual values of each output. We get the
# prediction probabilities by passing it through an instance of the nn.Softmax module.

X = torch.rand(1, 28, 28, device + device)
logits = model(x)
pred_probab = nn.Softmax(dim = 1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

# Model Layers
