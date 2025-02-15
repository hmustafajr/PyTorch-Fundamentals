# Preparing Data for Training (DataLoaders)
# the dataset retrieves our dataset's features and laels one sample at a time. when training a moedl, we typically want to
# pass samples in "minibatches", reshuffle the data at every epoch to reduce model overfitting, and use python's
# multiprocessing to speed up data retieval

# dataloader is an iterable that abstracts the complexity for us in an easy API

from torch.utils.data import DataLoader
train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)
