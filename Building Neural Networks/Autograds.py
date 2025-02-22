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

# Tensors, Functions and Computational graph
# In this network, w and b are parameters, which need to be optimized. We need to be able to compute the gradients of loss
# function with respect to those variables. In order to do that, we set the requires_grad property.
#
# ::Note:: We can set the value of requirea_grad when creating a tensor, or later by using x.requires_grad_(True) method
#
# A function that we wapply to tensors to construct computational graph is in fact an object of class Function. This object
# knows how to compute the function in the forward direction, and also how to compute its derivative during the backward
# propagation step. A reference tot he backward propagation is stored in grad_fn property of a tensor.

print(f"Gradient function for z = {z.grad_fn}")
print(f" Gradient function for loss = {loss.grad_fn}")
# Out
# Gradient function for z = <AddBackward0 object at 0x7f49adabc820>
# Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward0 object at 0x7f49adabece0>

# Computing Gradients
# To optimize weights of parameters in the neural network, we need to compute the derivatives of our loss function with
# respect to parameters, namely, we need 
# :math:`\frac{\partial loss}{\partial w}` and :math:`\frac{\partial loss}{\partial b}`
# under some fixed values of x and y. To compute those derivatives, we call loss.backward(), and then retrieve the values
# from w.grad and b.grad:

loss.backward()
print(w.grad)
print(b.grad)

# Output
# tensor([[0.3313, 0.0626, 0.2530],
#        [0.3313, 0.0626, 0.2530],
#        [0.3313, 0.0626, 0.2530],
#        [0.3313, 0.0626, 0.2530],
#        [0.3313, 0.0626, 0.2530]])
# tensor([0.3313, 0.0626, 0.2530])

# ::Note:: We can only obtain the grad properties for the leaf nodes of the computational graph, which have requires_grad
# property set to True. For all other nodes in the graph, gradients will not be available.

# ::Note:: we can only perform gradient calculations using backward once on a given graph, for performance reasons. if we
# need to do several backward calls on the same graphh, we need to pass retain_graph = True to the backward call.

