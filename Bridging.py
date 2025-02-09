# Tensors on the CPU and NumPy arrays can share memory locations and changing one will change the other

# Tensor to NumPy array
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
# Output
#   t: tensor([1., 1., 1., 1., 1.])
#   n: [1. 1. 1. 1. 1.]
