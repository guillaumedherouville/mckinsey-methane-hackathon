import os
import sys
import datetime
import argparse
from argparse import Namespace

import torch
import numpy as np
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.append(".")
from utils.data import PlumesDataset


class TorchTrainer:
    """The trainer class, which encapsulates PyTorch model training logic.

    @param model: The PyTorch model.
    @param train_loader: The train dataloader.
    @param test_loader: The test dataloader.
    @param criterion: The loss function type.
    @param optimizer: The optimizer used to update model's weights.
    @param device: The device, which perform computations on.
    @param train_config: The namespace with the cmd arguments. Is used to control training process.
    """

    def __init__(self, model: Module, train_loader: DataLoader, test_loader: DataLoader,
                 criterion: Module, optimizer: Optimizer, device: str, train_config: Namespace) -> None:
        self.model = model

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.criterion = criterion
        self.optimizer = optimizer

        self.device = device

        self.train_config = train_config

        self.metrics = dict()

    def dump_model(self, **tags) -> None:
        """Dump model weights.

        The weight's filename includes current timestamp, train-config params, metrics and tags.

        @param tags: Additional information to include in the weights' filename.
        @return: Nothing.
        """
        if not os.path.isdir(self.train_config.save_weights_dir):
            os.mkdir(self.train_config.save_weights_dir)

        tags_info = "--".join([f"{k}:{v}" for k, v in tags.items()])
        metrics_info = "--".join([f"{k}:{v}" for k, v in self.metrics.items()])
        weights_name = os.path.join(self.train_config.save_weights_dir, f"{datetime.datetime.now()}--"
                                                                        f"{self.train_config.model_name}--"
                                                                        f"pretrained:{self.train_config.pretrained}--"
                                                                        f"batch-size:{self.train_config.batch_size}--"
                                                                        f"augmented:{self.train_config.augment}--"
                                                                        f"{tags_info}--"
                                                                        f"{metrics_info}")
        torch.save(self.model.state_dict(), weights_name)

    def train_epoch(self) -> None:
        """Perform train epoch.

        @return: Nothing.
        """
        # For the metrics.
        targets = list()
        predicted_probs = list()
        train_set_size = len(self.train_loader.dataset)

        # Train epoch.
        self.model.train()
        for batch_idx, (x, y_truth) in enumerate(self.train_loader):
            # Compute loss on a given batch.
            x = x.to(self.device)
            y_pred = self.model(x)
            y_truth = y_truth.unsqueeze(1).to(self.device)
            batch_loss = self.criterion(y_pred, y_truth)

            # Compute predicted probability.
            y_pred = sigmoid(y_pred)

            # Update weights.
            batch_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Store targets and predictions for the metrics calculation.
            targets.extend(y_truth.cpu().detach().numpy().flatten().tolist())
            predicted_probs.extend(y_pred.cpu().detach().numpy().flatten().tolist())

            # Display loss.
            if batch_idx % 10 == 0:
                batch_loss, samples_trained = batch_loss.item(), (batch_idx + 1) * len(x)
                print(f"loss: {batch_loss:>5f}  [{samples_trained:>5d}/{train_set_size:>5d}]")

        # Metrics calculation.
        correct_predictions = (np.array(targets) == (np.array(predicted_probs) >= 0.5)).sum()
        accuracy = correct_predictions / train_set_size
        print(f"\nTrain Accuracy: {(100 * accuracy):>0.2f}%")

        roc_auc = roc_auc_score(targets, predicted_probs)
        print(f"Train ROC-AUC score: {(100 * roc_auc):>0.2f}%")

        # Track latest metrics.
        self.metrics["train_accuracy"] = accuracy
        self.metrics["train_roc_auc"] = roc_auc

    def test_epoch(self) -> None:
        """Perform test epoch.

        @return: Nothing.
        """
        # For the metrics.
        targets = list()
        predicted_probs = list()
        test_set_size = len(self.test_loader.dataset)

        # For the loss.
        test_loss = 0
        num_test_batches = len(self.test_loader)

        # Test epoch.
        self.model.eval()
        with torch.no_grad():
            for x, y_truth in self.test_loader:
                x = x.to(self.device)
                y_pred = self.model(x)
                y_truth = y_truth.unsqueeze(1).to(self.device)
                test_loss += self.criterion(y_pred, y_truth)

                targets.extend(y_truth.cpu().detach().numpy().flatten().tolist())
                predicted_probs.extend(y_pred.cpu().detach().numpy().flatten().tolist())

        # Metrics calculation.
        correct_predictions = (np.array(targets) == (np.array(predicted_probs) >= 0.5)).sum()
        accuracy = correct_predictions / test_set_size
        print(f"\nTest Accuracy: {(100 * accuracy):>0.2f}%")

        roc_auc = roc_auc_score(targets, predicted_probs)
        print(f"Test ROC-AUC score: {(100 * roc_auc):>0.2f}%")

        test_loss /= num_test_batches
        print(f"Test Loss: {test_loss:>5f}\n")

        # Track latest metrics.
        self.metrics["test_accuracy"] = accuracy
        self.metrics["test_roc_auc"] = roc_auc


def _get_device() -> str:
    """Get the fastest available device.

    Determines, which device is available on the host and returns the fastest one.
    """
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _init_model(train_config: Namespace) -> Module:
    """Get a model instance by its name.

    Instantiate a model of a class, given by the 'train_config.model_name'.

    @param train_config: The training pipeline config. Controls the training process.
    @return: The PyTorch model instance.
    """
    model_class = getattr(__import__("architectures"), train_config.model_name)
    model_instance = model_class(pretrained=train_config.pretrained).double()
    return model_instance


def main(train_config: Namespace) -> None:
    """
    @param train_config: The training pipeline config. Controls the training process.
    @return: Nothing.
    """
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
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(train_config.batch_size)
    )

    # Dynamically import model by its name.
    device = _get_device()
    model = _init_model(train_config)
    model.to(device)

    # Define loss criterion and optimizer.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config.learning_rate))

    # Model training.
    torch_trainer = TorchTrainer(model, train_loader, test_loader, criterion, optimizer, device, train_config)
    for epoch in range(int(train_config.epochs)):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        torch_trainer.train_epoch()
        torch_trainer.test_epoch()

        if (epoch + 1) % int(train_config.save_weights_freq) == 0:
            torch_trainer.dump_model(epoch=epoch+1)

    print("\nDone!")


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

    main(parser.parse_args())
