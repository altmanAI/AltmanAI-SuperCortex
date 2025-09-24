"""Core orchestrator for AltmanAI‑SuperCortex.

This class serves as a central coordinator for model training, saving, and inference. It can
be expanded to manage multiple models, datasets, and more complex workflows.
"""
from typing import Optional
import torch
from torch.utils.data import DataLoader

from .models import SimpleNet
from .training import generate_synthetic_data, train_model, save_model


class AltmanAISuperCortex:
    """Central orchestrator for training and inference within the AltmanAI ecosystem."""

    def __init__(self) -> None:
        self.model: Optional[SimpleNet] = SimpleNet()

    def train(self, num_samples: int = 1000, epochs: int = 5, lr: float = 0.001) -> None:
        """Train the internal model using synthetic data.

        Args:
            num_samples: Number of synthetic samples to generate.
            epochs: Number of training epochs.
            lr: Learning rate for optimization.
        """
        dataset = generate_synthetic_data(num_samples=num_samples)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        if self.model is None:
            self.model = SimpleNet()
        self.model = train_model(self.model, dataloader, epochs=epochs, lr=lr)

    def save(self, path: str = "models/simple_net.pt") -> None:
        """Save the trained model to disk.

        Args:
            path: Destination file for the saved model parameters.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        save_model(self.model, path)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Run inference on a batch of inputs.

        Args:
            X: A tensor of shape (N, input_dim) containing input samples.

        Returns:
            A tensor of predicted class indices.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            return torch.argmax(logits, dim=1)
