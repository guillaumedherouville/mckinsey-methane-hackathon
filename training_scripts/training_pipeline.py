import os
import sys
import datetime

sys.path.append(".")

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.data import PlumesDataset


def _define_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return device


def _train_epoch(model, dataloader, criterion, optimizer):
    train_set_size = len(dataloader.dataset)
    correct_predictions = 0

    model.train()
    for batch_idx, (X, y_truth) in enumerate(dataloader):
        # Compute loss on a given batch.
        y_truth = y_truth.unsqueeze(1)
        y_pred = model(X)
        batch_loss = criterion(y_pred, y_truth)

        # Update weights.
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Accuracy.
        correct_predictions += (y_truth == (y_pred >= 0.5)).type(torch.float).sum().item()

        # Display loss.
        if batch_idx % 10 == 0:
            batch_loss, samples_trained = batch_loss.item(), (batch_idx + 1) * len(X)
            print(f"loss: {batch_loss:>5f}  [{samples_trained:>5d}/{train_set_size:>5d}]")

    accuracy = correct_predictions / train_set_size
    print(f"Train Accuracy: {(100 * accuracy):>0.2f}%")

    return accuracy


def _test_epoch(model, dataloader, criterion):
    test_set_size = len(dataloader.dataset)
    num_test_batches = len(dataloader)
    test_loss, correct_predictions = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y_truth in dataloader:
            y_truth = y_truth.unsqueeze(1)
            y_pred = model(X)
            test_loss += criterion(y_pred, y_truth)
            correct_predictions += (y_truth == (y_pred >= 0.5)).type(torch.float).sum().item()

    test_loss /= num_test_batches
    accuracy = correct_predictions / test_set_size
    print(f"Test Accuracy: {(100 * accuracy):>0.2f}%\n"
          f"Test Loss: {test_loss:>5f}\n")

    return accuracy


def _model_dump(model, train_accuracy, test_accuracy, train_config, **kwargs):
    if not os.path.isdir(train_config.save_weights_dir):
        os.mkdir(train_config.save_weights_dir)

    torch.save(model.state_dict(), os.path.join(train_config.save_weights_dir, f"{datetime.datetime.now()}--"
                                                                               f"train-accuracy:{train_accuracy}--"
                                                                               f"test-accuracy:{test_accuracy}--"
                                                                               f"{train_config.model_name}--"
                                                                               f"pretrained:{train_config.pretrained}--"
                                                                               f"batch-size:{train_config.batch_size}--"
                                                                               f"augmented:{train_config.augment}--"
                                                                               f"epoch:{kwargs['epoch']}"))


def main(train_config):
    device = _define_device()

    train_dataset = PlumesDataset(
        root_dir=train_config.root_dir,
        is_train=True,
        augment=train_config.augment
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.batch_size),
        shuffle=True
    )

    test_dataset = PlumesDataset(
        root_dir=train_config.root_dir,
        is_train=False,
        augment=False  # We never augment test-data.
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_config.batch_size)
    )

    # Dynamically import model by its name.
    module = __import__("model")
    model_class = getattr(module, train_config.model_name)
    model = model_class(pretrained=train_config.pretrained).double().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.learning_rate))
    criterion = nn.BCEWithLogitsLoss()

    # Launch model training.
    best_test_accuracy = .0

    for epoch in range(int(train_config.epochs)):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_accuracy = _train_epoch(model, train_loader, criterion, optimizer)
        test_accuracy = _test_epoch(model, test_loader, criterion)

        if test_accuracy >= best_test_accuracy:
            best_test_accuracy = test_accuracy
            _model_dump(model, train_accuracy, test_accuracy, train_config, epoch=epoch)

        if epoch % int(train_config.save_weights_freq) == 0:
            _model_dump(model, train_accuracy, test_accuracy, train_config, epoch=epoch)

    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--learning-rate", help="Learning rate.")
    parser.add_argument("--batch-size", help="Batch size.")
    parser.add_argument("--epochs", help="# Of training epochs")
    parser.add_argument("--save-weights-dir", default="./weights", help="The directory to save weights.")
    parser.add_argument("--save-weights-freq", default=5, help="How frequent to dump model weights.")
    parser.add_argument("--root-dir", default="./data", help="The root directory with data.")
    parser.add_argument("--augment", action="store_true", help="Whether to augment training set.")

    args = parser.parse_args()
    main(args)
