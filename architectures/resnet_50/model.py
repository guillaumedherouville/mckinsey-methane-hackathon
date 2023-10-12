import torch
from torch import nn
from torchvision.models import resnet50

from ..cls_head import ClassificationHead


class ResNet50(nn.Module):
    """ResNet50 model adaptation."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = resnet50(*args, **kwargs)
        self.model.fc = ClassificationHead(in_features=2048, n_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward step.

        @param x: The input tensor.
        @return: The output logit.
        """
        logit = self.model(x)
        return logit


if __name__ == '__main__':
    import numpy as np
    dummy_input = torch.from_numpy(np.random.random(size=(2, 3, 64, 64)))

    model = ResNet50(pretrained=False).double()
    print(model(dummy_input))
    print(model)
