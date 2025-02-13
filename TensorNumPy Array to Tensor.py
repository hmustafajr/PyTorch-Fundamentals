n = np.ones(5)
t = torch.from_numpy(n)

# changes in the NumPy array reflects in the tensor
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n {n}")

# output
#   t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
#   n: [2. 2. 2. 2. 2.]
