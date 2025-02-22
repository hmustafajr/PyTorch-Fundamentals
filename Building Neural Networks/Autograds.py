# When training neural networks, the most frequently used algorithm is back propogation. In this algorithm,
# parameters (model weights) are adjusted according to the gradient of the loss function with respect to the given parameter.
#
# To compare those gradients, PyTorch has a buil-in differentiation engine torch.autograd. It supports automatic computation
# of gradient for any computational graph.
#
# Consider the most simple single layer neural network that has input x, parameters w and b, and some loss function. It can
# be defined in PyTorch in the following manner:

import torch
x = torch.ones(5) # tensor input
y = torch.zeros(3) # expected output
w = torch.rand(5, 3, requires_grad = true)
b = torch.rand(3, requires_grad = true)
z = torch.matmul(x, w)+ b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
