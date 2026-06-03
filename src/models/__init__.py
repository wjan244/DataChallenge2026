import torch.nn as nn
from src.models.scratch_cnn import ConvNet, ResNet18, EfficientNet

CUSTOM_MODELS = {"convnet": ConvNet, "resnet18": ResNet18, "efficientnet": EfficientNet}
