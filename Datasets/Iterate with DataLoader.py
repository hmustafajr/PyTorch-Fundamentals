# we have loaded that dataset into the DataLoader and can iterate through the data as needed. each iteration
# below returns a batch of train_features and trainf_labels (containing batch_size = 64 features and labels respectively)
# because we specified shuffle = True, after we iterate over all batches the data is shuffled (for finer-grained control
# over the data loading order, take a look at Samplers).

# displat image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imgshow(img, cmap = "gray")
plt.show()
print(f"Label: {label}")
