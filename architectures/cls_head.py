from torch import nn
import torch.nn.functional as F


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
