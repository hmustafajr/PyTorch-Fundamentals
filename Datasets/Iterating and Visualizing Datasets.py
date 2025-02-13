# we can index Datasets manually like a list: training_data[index]. we user matplotlib to visualize some samples

labels_map = {
  0: "t-shirt",
  1: "trousers",
  2: "pullover",
  3: "dress",
  4: "coat",
  5: "sandal",
  6: "shirts",
  7: "sneaker",
  8: "bag",
  9: "ankle boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows +1):
  sample_idx = torch.randint(len(training_data), size = (1,)).item()
  img, label = training_data[sample_idx]
  figure.add_subplot(rows, cols, i)
  plt.axis("off")
  plt.imshow(img.squeeze(), cmap="gray")
plt.show()
