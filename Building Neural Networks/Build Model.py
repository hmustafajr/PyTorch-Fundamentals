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
# We'll break down the layers in the FashionMNIST model for illustration. We'll take a sample minibatch of a few images sized
# 28 x 28 and see what happens when we pass it through the network.

input_image = torch.rand(2, 28, 28)
print(input_image.size())

#Out
# torch.size([3, 28, 28])

# Flatten
# We initialize the nn.Flatten layer to convert each 2 dimensional image into a contiguous array of 784 pixel values
# the minibatch dimension (at dim=0) is maintained.

flatten = nn.Flatten()
flat_image = flatten(inout_image)
print(flat_image.size())

# Out
# torch.Size([3, 784])

# Linear
# The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.

layer1 = nn.Linear(in_features = 28* 28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# Out
# torch.size([3, 20])

# ReLU
# Non-linear activations are what create the complex mapping between the model's inputs and outputs. They are applied
# after linear transformations to introduce nonlinearity, helping nueral networks learn a wide variety of phenomena.
# We'll use the nn.ReLU between our linear layers, but there are other activations to introduce non-linearity.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# Out
# Before ReLU: tensor([[ 0.4158, -0.0130, -0.1144,  0.3960,  0.1476, -0.0690, -0.0269,  0.2690,
#         0.1353,  0.1975,  0.4484,  0.0753,  0.4455,  0.5321, -0.1692,  0.4504,
#         0.2476, -0.1787, -0.2754,  0.2462],
#         [ 0.2326,  0.0623, -0.2984,  0.2878,  0.2767, -0.5434, -0.5051,  0.4339,
#         0.0302,  0.1634,  0.5649, -0.0055,  0.2025,  0.4473, -0.2333,  0.6611,
#         0.1883, -0.1250,  0.0820,  0.2778],
#         [ 0.3325,  0.2654,  0.1091,  0.0651,  0.3425, -0.3880, -0.0152,  0.2298,
#         0.3872,  0.0342,  0.8503,  0.0937,  0.1796,  0.5007, -0.1897,  0.4030,
#         0.1189, -0.3237,  0.2048,  0.4343]], grad_fn=<AddmmBackward0>)

# After ReLU: tensor([[0.4158, 0.0000, 0.0000, 0.3960, 0.1476, 0.0000, 0.0000, 0.2690, 0.1353,
#         0.1975, 0.4484, 0.0753, 0.4455, 0.5321, 0.0000, 0.4504, 0.2476, 0.0000,
#         0.0000, 0.2462],
#         [0.2326, 0.0623, 0.0000, 0.2878, 0.2767, 0.0000, 0.0000, 0.4339, 0.0302,
#         0.1634, 0.5649, 0.0000, 0.2025, 0.4473, 0.0000, 0.6611, 0.1883, 0.0000,
#         0.0820, 0.2778],
#         [0.3325, 0.2654, 0.1091, 0.0651, 0.3425, 0.0000, 0.0000, 0.2298, 0.3872,
#         0.0342, 0.8503, 0.0937, 0.1796, 0.5007, 0.0000, 0.4030, 0.1189, 0.0000,
#         0.2048, 0.4343]], grad_fn=<ReluBackward0>)

# Sequential
# nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined.
# You can use sequential containers to put together a quick-network like seq_modules.

seq_modules = nn.Sequesntial(
  flatten,
  layer1,
  nnReLU(),
  nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# Softmax
# The last linear layer of the neural network returns logits-raw values in [-infinity, infinity] - which are passed to
# the nn.Softmax module. The logits are scaled to values [0, 1] representing the model's predicted probabilities for
# each class. dim parameter indicates the dimension along which the values must sum to 1

softmax = nn.Softmac(dim = 1)
pred_softmax = softmax(logits)
