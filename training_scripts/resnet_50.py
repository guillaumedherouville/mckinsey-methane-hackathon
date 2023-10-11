import torch.cuda
from torch import nn
from torch.utils.data import DataLoader
from model.resnet_50.model import ResNet50

from utils.convnext_dataloader import CustomDataset

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

learning_rate = 1e-3
batch_size = 16
epochs = 40

train_dataset = CustomDataset(
    root_dir="../data/dataset_split/train",
    is_train=True,
    n_channels=3
)

test_dataset = CustomDataset(
    root_dir="../data/dataset_split/test",
    is_train=False,
    n_channels=3
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)



loss_fn = nn.BCEWithLogitsLoss()
model = ResNet50(pretrained=False)
model.double()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_loop():
    size = len(train_dataloader.dataset)
    correct = 0

    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        # Compute prediction and loss
        y = y.unsqueeze(1)
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += ((pred >= 0.5) == y).type(torch.float).sum().item()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    print(f"Train Accuracy: {(100*correct):>0.2f}%")


def test_loop():
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in test_dataloader:
            y = y.unsqueeze(1)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred >= 0.5) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop()
    test_loop()
print("Done!")
