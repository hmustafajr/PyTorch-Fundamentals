# If you have a single element tensor, for example by aggregating all values of a tensot into one value,
# you can convert it to a python numberical value using item()
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# output
# 12.0 <class 'flaot'>
