# __getitem__
# the __getitem__ function loads and retursn a sample from the dataset at given index idx. based on the index, it identifies
# the image's location on a disk, converts that to a tensor using read-image, retrieves the corresponding label from the csv
# data in self.imf_labels, calls the transform functions on them where applicable, then returns the tesnor image and labels
# in a tuple

def __getitem__(self, idx):
  img_path = os.path.join(self,img_dir, self.img_labels.iloc[idx, 0])
  image = read_image(img_path)
  label = self.img_labels.iloc[idx, 1]
  if self.transform:
    image = self.transform(image)
  if self.target_transform:
    label = self.target_transform(label)
  return image, label
