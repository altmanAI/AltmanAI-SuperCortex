# AltmanAI‑SuperCortex

AltmanAI‑SuperCortex serves as a foundational “central nervous system” for AltmanAI. It includes a minimal machine‑learning pipeline using PyTorch and a central orchestrator class that can be extended to manage models, training, evaluation, and inference.

## Features

- **Modular architecture**: code organized into the `cortex` package with separate modules for models, training, and orchestration.
- **Simple training example**: trains a fully connected neural network on synthetic data to illustrate the workflow.
- **Command‑line interface**: run training or prediction from the command line via `main.py`.
- **Extensible**: designed to be extended with real datasets, more complex models, and additional tasks.

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**

   ```bash
   python main.py train --samples 1000 --epochs 5
   ```

   This trains a simple model on synthetic data and saves it under `models/simple_net.pt`.

3. **Run a prediction**

   ```bash
   # Create a sample tensor with shape (10, 784) and save it
   python -c "import torch; torch.save(torch.rand(10, 784), 'sample.pt')"
   python main.py predict --input sample.pt
   ```

## Next steps

- Replace the synthetic data generator with real datasets (e.g., MNIST) in `cortex/training.py`.
- Introduce more sophisticated models and training strategies.
- Expand the orchestrator to manage multiple models, datasets, and tasks across the AltmanAI ecosystem.
