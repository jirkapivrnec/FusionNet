from torchviz import make_dot
import torch

from FusionNet import FusionNet

# Instantiate your model
model = FusionNet(num_classes=10)

# Create dummy input to pass through the model
x = torch.randn(1, 3, 32, 32)

# Get the model output
y = model(x)

# Visualize the model architecture
dot = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
dot.format = 'png'
dot.render('fusionnet_architecture')
