# Now that we have a model and data it's time to train, validate and test our model by optimizing its parameters
# on our data. Training a model is an iterative process, in each iteration the model makes a guess about the output
# calculates the error in its guess(loss), collects the derivatives of the error with respect to it's parameters
# and optimizes these parameters using gradient descent.

# Prerequisite Code
# We'll load code from the Datasets & DataLoaders and Buld Model

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transform import ToTensor

training_data = datasets.FashionMNIST(
  root = "data",
  train = true,
  download = true,
  transform = ToTensor()
)

train_dataloader = DateLoader(training_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.linear_relu_stack = nn.Sequential(
      nn.linear(28 * 28, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 10),
    )
def forward(self, x):
  x = self.flatten(x)
  logits = self.linear_relu_stack(x)
  return logits
model = NeuralNetwork()

# Out
#   0%|          | 0.00/26.4M [00:00<?, ?B/s]
#  0%|          | 65.5k/26.4M [00:00<01:12, 361kB/s]
#  1%|          | 229k/26.4M [00:00<00:38, 679kB/s]
#  3%|3         | 885k/26.4M [00:00<00:10, 2.52MB/s]
#  7%|7         | 1.93M/26.4M [00:00<00:05, 4.09MB/s]
# 23%|##3       | 6.19M/26.4M [00:00<00:01, 14.3MB/s]
# 38%|###7      | 9.93M/26.4M [00:00<00:00, 17.3MB/s]
# 57%|#####7    | 15.1M/26.4M [00:01<00:00, 25.7MB/s]
# 71%|#######1  | 18.9M/26.4M [00:01<00:00, 24.2MB/s]
# 91%|#########1| 24.1M/26.4M [00:01<00:00, 30.7MB/s]
# 100%|##########| 26.4M/26.4M [00:01<00:00, 19.3MB/s]
#
#  0%|          | 0.00/29.5k [00:00<?, ?B/s]
# 100%|##########| 29.5k/29.5k [00:00<00:00, 324kB/s]
#
#  0%|          | 0.00/4.42M [00:00<?, ?B/s]
#  1%|1         | 65.5k/4.42M [00:00<00:12, 361kB/s]
#  4%|4         | 197k/4.42M [00:00<00:05, 767kB/s]
# 10%|9         | 426k/4.42M [00:00<00:03, 1.07MB/s]
# 36%|###6      | 1.61M/4.42M [00:00<00:00, 4.26MB/s]
# 79%|#######8  | 3.47M/4.42M [00:00<00:00, 7.09MB/s]
# 100%|##########| 4.42M/4.42M [00:00<00:00, 6.05MB/s]
#
#  0%|          | 0.00/5.15k [00:00<?, ?B/s]
# 100%|##########| 5.15k/5.15k [00:00<00:00, 40.5MB/s]

# Hyperparameters
# Hyperparameters are adjustable parameters that let you control the model optimization process. Different hyperparameter
# values can impact model training and convergence rates.
# We define the following hyperparameters for training
# - Number of Epochs - the number times to iterate over the dataset
# - Batch Size - the number of data samples propagated through the network before the parameters are updated
# - Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed,
# while large values may result in unpredictable behavior during training.

learning_rate = 1e-3
batch_size = 64
epochs = 5

# Optimization Loop
# Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each iteration
# of the optimization loop is called an epoch.
# Each epoch consists of two main parts:
# - The Train Loop - iterate over the traininf dataset and try to converge to optimal parameters.
# - The Validation / Test Loop - iterate over the test dataset to check if model performance is improving.

# We'll briefly familiarize ourselves with some of the concepts used in the training loop.

# Loss Function
# When presented with some training data, our untrained network is likely not to give the correct answer. Loss Function
# measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we want
# to minimize during training. To calculate the loss we make a prediction using the inputs of our given data sample and 
# compare it against the true data label value.
#
# Common loss dunctions include nn.MSELoss (Mean Square Error) for regression tasks, and nnNLLLoss (Negative Log Likelihood(
# for classification. nnCrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
#
# We pass our model's output logits to nn.CrossEntropyLoss, which will normalize the logits and compute the prediction error.

# Inititalize the loss function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
#
#
