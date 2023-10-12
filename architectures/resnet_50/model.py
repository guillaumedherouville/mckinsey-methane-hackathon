import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50


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


class ResNet50(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = resnet50(*args, **kwargs)
        self.model.fc = ClassificationHead(in_features=2048, n_classes=1)

    def forward(self, x):
        logit = self.model(x)
        return logit


if __name__ == '__main__':
    import numpy as np
    dummy_input = torch.from_numpy(np.random.random(size=(2, 3, 64, 64)))

    model = ResNet50(pretrained=False).double()
    print(model(dummy_input))
    print(model)
