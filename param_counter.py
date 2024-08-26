import torch
import torch.nn as nn

from FusionNet import FusionNet

# Instantiate the model
model = FusionNet()

# Calculate the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(total_params)