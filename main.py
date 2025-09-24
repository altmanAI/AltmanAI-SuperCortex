"""Command‑line interface for AltmanAI‑SuperCortex.

This script exposes training and prediction capabilities via a simple CLI. It relies on
`cortex.core.AltmanAISuperCortex` for the underlying functionality.
"""
import argparse
import torch
from cortex.core import AltmanAISuperCortex


def main() -> None:
    parser = argparse.ArgumentParser(description="AltmanAI‑SuperCortex CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub‑commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model with synthetic data")
    train_parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples")
    train_parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Run inference on a saved input tensor (.pt)")
    predict_parser.add_argument("--input", type=str, required=True, help="Path to a .pt file containing input data")

    args = parser.parse_args()
    cortex = AltmanAISuperCortex()

    if args.command == "train":
        cortex.train(num_samples=args.samples, epochs=args.epochs, lr=args.lr)
        cortex.save()
        print("Training completed and model saved to models/simple_net.pt")
    elif args.command == "predict":
        X = torch.load(args.input)
        preds = cortex.predict(X)
        print("Predictions:", preds)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
