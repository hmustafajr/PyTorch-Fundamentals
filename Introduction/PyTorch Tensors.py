import torch

# Let's try out some basic tensor manipulation
z = torch.zeros(5, 3)
print(z)
print(z.dtype)
#
# Output is a 5 by 3 matrix full of zeros
tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
torch.float32 # The default type is a 32 bit point float

# What if we don't want a float but an integer?
i = torch.ones((5, 3), dtype = torch.int16)
print(i)
#
# Output is a 5 by 3 matrix full of ones
tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.int16) # Observe that this is now an integer
#
# It's common to initialize learning weights randomly, often with a specific seed for the PRNG for reproducibility
torch.manual_seed(1529)
r1 = torch.ran(2, 2)
print('Random Tensor:')
print(r1)

r2 = torch.rand_seed(2, 2)
print('Another Random Tensor')
print(r2)

torch.manual_seed(1529)
print('\nShould match r1:')
print(r3) # repeats values of r1 because of re-seed
#
# Output
A random tensor:
tensor([[0.9685, 0.2850],
        [0.7968, 0.0484]])
A different random tensor:
tensor([[0.2644, 0.7690],
        [0.5193, 0.1623]])
Should match r1:
tensor([[0.9685, 0.2850],
        [0.7968, 0.0484]])

# PyTorch tensors perform arithmetic operations intuitively. Tensors of similar shapes may be added, multiplied. etc.
# Operations with scalars are distributed over the tensor:
ones = torch.ones(2, 3)
print(ones)

twos = torch.ones(2, 3) * 2 # each element is multiplied by 2
print(twos)

threes = ones + twos # addition allowed because shapes are similar
print(threes) # tensors are added elementally
print(threes.shape) # this has the same dimensions as input tensors

r1 = torch.rand(2, 3)
r2 = torch.rand(2, 3)
#r3 = r1 + r2 #uncomment if you want a runtime error
