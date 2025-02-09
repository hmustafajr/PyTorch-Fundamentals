# In-place operations store the results into the operand. These are denoted by a _ suffix.
print(f" {tensor} \n")
tensor.add_(5)
print(tensor)

# In-place operations save some memory, but can be problematic when computing derivatives
# because of the immediate loss of history.
