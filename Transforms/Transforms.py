# Data doesn't arrive in a final processsed form that is required for training machine learning algorithms.
# Transforms is used to perform manipultion of the data making it suirable for training.

# TorchVision datasets have two parameters - transform to modify the features and - target_transform to modify
# the labels that accept callables containing the transformation logic. The torchvision.transforms module offers several
# commonly-used transforms out of the box.

# The FashionMNIST fetures are in PIL Image Format, and the labels are integers. For training the features need to be
# normalized tensors, and the labels as one-hot encoded tensors. To make the transformations, we use ToTensor and Lamda.

import torch
from torchvison import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST (
  root = "data",
  train = True,
  downlaod = True,
  transform = ToTenso(),
  target_transform = Lambda (lambda y: torch.zeros(10, dtype = torch.float).scatter_(0, torch.tensor(y), value = 1))
)
