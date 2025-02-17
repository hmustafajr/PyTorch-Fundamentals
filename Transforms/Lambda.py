# Lambda transforms apply any user-defined function. Here we define a function to turn the integer into a one-hot
# encoded tensor. First it creates a zero tensor of size 10 (the number of labels in our dataset) and calls
# scatter_ which assigns a value = 1 on the index as given by the label y.

target_transform = Lambda(lambda y: torch.zeros(
  10, dtype = torch.float).scatter_(dim = 0, index = torch.tensor(y), value = 1)
)
