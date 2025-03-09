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
# Optimization is the process of adjusting model parameters to reduce model error in each training step.
# Optimization algorithms define how this process is performed (in this example we use Stochastic Gradient Descent).
# All optimization logic is encapsulated in the optimizer object. Here, we use the SGD optimizer, additionally, there
# are many different optimizers available in PyTorch such as ADAM and RMSProp, that work better for differrent kinds of
# models and data.
#
# We initialize the optimizer by registering the model's parameters that need to be trained, and passing in the learning
# rate hyperparameter.

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# Inside the training loop, optimization happens in three steps:
# - Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up: to prevent
#   double counting, we explicitly zero them at each iteration.
# - Backpropogate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients collected in the
#   pass.

# Full Implementation
#
# We define train_loop that loops over our optimization code, and test_loop that evaluates the model's performance against
# our test data.

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  # Set the model to training mode - important for batch normilization and dropout layers
  # Unnecessary in this situation but added because it's a best practice
  model.train()
  for batch, (X, y) in enumerate(dataloader):
    # Compute prediction loss
    pred = model(X)
    loss = loss_fn(pred, y)
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if batch % 100 == 0:
      loss, current = loss.item(), batch * batch_size + len(X)
      print(f"loss: {loss:>7f} [{current:>5d}/{size:5d}]")
      
  def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normilization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

# Evaluating the model with torch.no.grad() ensures that no gradients are computed during test mode
# Also serves to reduce unneccessary gradient computations and memory usage for tensors with requires_grad=True
   with torch.no_grad():
     for X, y in dataloader:
       pred = model(X)
       test_loss += loss_fn(pred, y).item()
       correct += (pred.argmax(1) == y).type(torch.float).sum().item()

   test_loss /= num_batches
   correct /= size
   print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

# We initialize the loss function and optimizer, and pass it to train_loop and test_loop. Feel free to increase the number
# of epochs to track the model's improving performance.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

epochs = 10
for t in range(epochs):
  print(f"Epoch {t+1}\n---------")
  train_loop(train_dataloader, model, loss_fn, optimizer)
  test_loop(test_dataloader, model, loss_fn)
print("Done!")

# Out
Epoch 1
-------------------------------
loss: 2.298730  [   64/60000]
loss: 2.289123  [ 6464/60000]
loss: 2.273286  [12864/60000]
loss: 2.269406  [19264/60000]
loss: 2.249603  [25664/60000]
loss: 2.229407  [32064/60000]
loss: 2.227368  [38464/60000]
loss: 2.204261  [44864/60000]
loss: 2.206193  [51264/60000]
loss: 2.166651  [57664/60000]
Test Error:
 Accuracy: 50.9%, Avg loss: 2.166725

Epoch 2
-------------------------------
loss: 2.176750  [   64/60000]
loss: 2.169595  [ 6464/60000]
loss: 2.117500  [12864/60000]
loss: 2.129272  [19264/60000]
loss: 2.079674  [25664/60000]
loss: 2.032928  [32064/60000]
loss: 2.050115  [38464/60000]
loss: 1.985236  [44864/60000]
loss: 1.987887  [51264/60000]
loss: 1.907162  [57664/60000]
Test Error:
 Accuracy: 55.9%, Avg loss: 1.915486

Epoch 3
-------------------------------
loss: 1.951612  [   64/60000]
loss: 1.928685  [ 6464/60000]
loss: 1.815709  [12864/60000]
loss: 1.841552  [19264/60000]
loss: 1.732467  [25664/60000]
loss: 1.692914  [32064/60000]
loss: 1.701714  [38464/60000]
loss: 1.610632  [44864/60000]
loss: 1.632870  [51264/60000]
loss: 1.514263  [57664/60000]
Test Error:
 Accuracy: 58.8%, Avg loss: 1.541525

Epoch 4
-------------------------------
loss: 1.616448  [   64/60000]
loss: 1.582892  [ 6464/60000]
loss: 1.427595  [12864/60000]
loss: 1.487950  [19264/60000]
loss: 1.359332  [25664/60000]
loss: 1.364817  [32064/60000]
loss: 1.371491  [38464/60000]
loss: 1.298706  [44864/60000]
loss: 1.336201  [51264/60000]
loss: 1.232145  [57664/60000]
Test Error:
 Accuracy: 62.2%, Avg loss: 1.260237

Epoch 5
-------------------------------
loss: 1.345538  [   64/60000]
loss: 1.327798  [ 6464/60000]
loss: 1.153802  [12864/60000]
loss: 1.254829  [19264/60000]
loss: 1.117322  [25664/60000]
loss: 1.153248  [32064/60000]
loss: 1.171765  [38464/60000]
loss: 1.110263  [44864/60000]
loss: 1.154467  [51264/60000]
loss: 1.070921  [57664/60000]
Test Error:
 Accuracy: 64.1%, Avg loss: 1.089831

Epoch 6
-------------------------------
loss: 1.166889  [   64/60000]
loss: 1.170514  [ 6464/60000]
loss: 0.979435  [12864/60000]
loss: 1.113774  [19264/60000]
loss: 0.973411  [25664/60000]
loss: 1.015192  [32064/60000]
loss: 1.051113  [38464/60000]
loss: 0.993591  [44864/60000]
loss: 1.039709  [51264/60000]
loss: 0.971077  [57664/60000]
Test Error:
 Accuracy: 65.8%, Avg loss: 0.982440

Epoch 7
-------------------------------
loss: 1.045165  [   64/60000]
loss: 1.070583  [ 6464/60000]
loss: 0.862304  [12864/60000]
loss: 1.022265  [19264/60000]
loss: 0.885213  [25664/60000]
loss: 0.919528  [32064/60000]
loss: 0.972762  [38464/60000]
loss: 0.918728  [44864/60000]
loss: 0.961629  [51264/60000]
loss: 0.904379  [57664/60000]
Test Error:
 Accuracy: 66.9%, Avg loss: 0.910167

Epoch 8
-------------------------------
loss: 0.956964  [   64/60000]
loss: 1.002171  [ 6464/60000]
loss: 0.779057  [12864/60000]
loss: 0.958409  [19264/60000]
loss: 0.827240  [25664/60000]
loss: 0.850262  [32064/60000]
loss: 0.917320  [38464/60000]
loss: 0.868384  [44864/60000]
loss: 0.905506  [51264/60000]
loss: 0.856353  [57664/60000]
Test Error:
 Accuracy: 68.3%, Avg loss: 0.858248

Epoch 9
-------------------------------
loss: 0.889765  [   64/60000]
loss: 0.951220  [ 6464/60000]
loss: 0.717035  [12864/60000]
loss: 0.911042  [19264/60000]
loss: 0.786085  [25664/60000]
loss: 0.798370  [32064/60000]
loss: 0.874939  [38464/60000]
loss: 0.832796  [44864/60000]
loss: 0.863254  [51264/60000]
loss: 0.819742  [57664/60000]
Test Error:
 Accuracy: 69.5%, Avg loss: 0.818780

Epoch 10
-------------------------------
loss: 0.836395  [   64/60000]
loss: 0.910220  [ 6464/60000]
loss: 0.668506  [12864/60000]
loss: 0.874338  [19264/60000]
loss: 0.754805  [25664/60000]
loss: 0.758453  [32064/60000]
loss: 0.840451  [38464/60000]
loss: 0.806153  [44864/60000]
loss: 0.830360  [51264/60000]
loss: 0.790281  [57664/60000]
Test Error:
 Accuracy: 71.0%, Avg loss: 0.787271

Done!
