"""Model definitions for AltmanAI‑SuperCortex.

Currently includes a simple fully connected neural network for demonstration purposes.
"""
import torch.nn as nn


class SimpleNet(nn.Module):
    """A minimal feedforward neural network with one hidden layer.

    Args:
        input_dim: Dimensionality of the input features.
        hidden_dim: Number of neurons in the hidden layer.
        output_dim: Number of output classes.
    """

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Flatten input except for batch dimension
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
