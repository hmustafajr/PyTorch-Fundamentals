# __init__
# the __init__ function is run once when instantiating the dataset object. Here we initialize the directory containing the
# images,the annotations file, and both transforms. More below
#  the labels csv file looks like:
#    tshirt1.jpg, 0
#    tshirt2.jpg, 0
#    ......
#    ankleboot999.jpg, 9

def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
  self.img_labels = pd.read_csv(annotations_file)
  self.img_dir = img_dir
  self.transform = transform
  self.target_transform = target_transform
