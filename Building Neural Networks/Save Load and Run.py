# In this part, we'll look at how to persist model state with saving, loading and running model predictions.
import torch
import torchvision.models as models

# Saving / Loading Model Weights
# PyTorch models store the learned parameters in an internal state dicstionary called state_dict. These can be persisted via
# the torch.save method

model - models.vgg16(weights= 'IMAGENET1K_v1')
torch.save(model.state_dict(), 'model_weights.pth')

# Out
# Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to /var/lib/ci-user/.cache/torch/hub/checkpoints/vgg16-397923af.pth
# 0%|          | 0.00/528M [00:00<?, ?B/s]
# 4%|3         | 20.8M/528M [00:00<00:02, 216MB/s]
# 8%|7         | 42.1M/528M [00:00<00:02, 221MB/s]
# 12%|#2        | 63.5M/528M [00:00<00:02, 222MB/s]

# To load model wieghts, you need to create an insurance of the same model first, and then load the parameters using
# load_state_dict() method
#
# In the code below, we set weights_only = True to limit the funcstions executed during unpicking to only those necessary
# for loading weights. Using weights_only = True is considred a best practice when loading weights.

model = models.vgg16() # we don't specify wewights i.e an untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only = True))
model.eval()

#Out
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

# Note - be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers
# to evaluation mode. Failing to do this will yeild inconsistent inference results.

# Saving / Loading Models w/ Shapes
# When loading model weights, we need to instantiate the model class first, because the class defines the structure
# of a network. We might want to save the structure of this class together with the model, in which case we pass model
# and not model.state_dict() to the saving function:

torch.save(model, 'model.pth')

# We can then load the model as demonstrated below
# Saving state_dict is considered the best practice. However we use weights_only = Fa;se because this involves the model,
# which is a legacy use case for torch.save.

model = torch.load('model.pth', weights_only = False),

# Note - this approach uses Python pickle module when serializing the model, thus it relies on the actural class definition
# to be available when loading the model
