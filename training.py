"""Training utilities for AltmanAI‑SuperCortex.

Provides functions to generate synthetic data, train models, and save trained weights.
Replace the synthetic data generator with real datasets to extend functionality.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .models import SimpleNet


def generate_synthetic_data(num_samples: int = 1000, input_dim: int = 784, num_classes: int = 10) -> TensorDataset:
    """Generate a synthetic dataset for demonstration.

    Args:
        num_samples: Number of samples to generate.
        input_dim: Dimensionality of each input sample.
        num_classes: Number of target classes.

    Returns:
        A TensorDataset containing random inputs and random labels.
    """
    X = torch.rand(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(X, y)


def train_model(model: SimpleNet, dataloader: DataLoader, epochs: int = 10, lr: float = 0.001) -> SimpleNet:
    """Train a neural network on a provided dataset.

    Args:
        model: The neural network model to train.
        dataloader: A DataLoader providing (input, label) batches.
        epochs: Number of passes over the entire dataset.
        lr: Learning rate for the optimizer.

    Returns:
        The trained model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        for X, y in dataloader:
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X.size(0)
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == y).sum().item()
            total += X.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total * 100.0
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return model


def save_model(model: SimpleNet, path: str) -> None:
    """Persist model weights to disk.

    Args:
        model: Trained neural network model.
        path: Output file path for saving the model state.
    """
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
