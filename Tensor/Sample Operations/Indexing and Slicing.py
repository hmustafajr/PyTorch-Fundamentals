tensor = torch.ones(4, 4)
print(f"First Row: {tensor[0]}")
print(f"First Column: {tensor[:, 0]}")
print(f"Last Column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

# Output example
#    First row: tensor([1., 1., 1., 1.])
#    First column: tensor([1., 1., 1., 1.])
#    Last column: tensor([1., 1., 1., 1.])
#    tensor([[1., 0., 1., 1.],
#           [1., 0., 1., 1.],
#           [1., 0., 1., 1.],
#           [1., 0., 1., 1.]])
