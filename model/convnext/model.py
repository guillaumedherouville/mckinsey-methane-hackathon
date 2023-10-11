import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import convnext_base


class ClassificationHead(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=1024)
        self.linear_2 = nn.Linear(in_features=1024, out_features=128)
        self.linear_3 = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.relu(x)

        x = self.linear_2(x)
        x = F.relu(x)

        logit = self.linear_3(x)
        return logit


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
