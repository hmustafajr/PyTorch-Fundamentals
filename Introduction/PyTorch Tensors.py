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
