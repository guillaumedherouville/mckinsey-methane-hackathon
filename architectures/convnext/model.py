import torch
from torch import nn
from torchvision.models import convnext_base

from ..cls_head import ClassificationHead


class ConvNeXtBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = convnext_base(*args, **kwargs)
        self.model.classifier[2] = ClassificationHead(in_features=1024, n_classes=1)

    def forward(self, x):
        logit = self.model(x)
        return logit


if __name__ == '__main__':
    import numpy as np
    dummy_input = torch.from_numpy(np.random.random(size=(2, 3, 64, 64)))

    model = ConvNeXtBase(pretrained=False).double()
    print(model(dummy_input))
    print(model)
