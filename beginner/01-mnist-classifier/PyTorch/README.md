# MNIST Digit Classifier — PyTorch

## Overview
A re-implementation of the MNIST digit classifier in PyTorch, originally
built with TensorFlow/Keras as the first project in this portfolio.
The goal was to learn PyTorch fundamentals by rebuilding a familiar problem
from scratch — writing the training loop manually instead of relying on
`model.fit()`.

## Results
| Metric | Score |
|--------|-------|
| Test accuracy | 99.58% |
| Framework | PyTorch |
| Original Keras result | 99.1% |
| Improvement | +0.48% |

## What's different from the Keras version

| Concept | Keras | PyTorch |
|---------|-------|---------|
| Model definition | `Sequential([...])` | Class inheriting `nn.Module` |
| Training | `model.fit()` | Manual loop with `loss.backward()` |
| Inference | `model.predict()` | `model(x)` with `torch.no_grad()` |
| Data loading | `ImageDataGenerator` | `Dataset` + `DataLoader` |
| GPU handling | Automatic | Explicit `.to(device)` |
| Dropout/BN modes | Automatic | Manual `model.train()` / `model.eval()` |

## Architecture
Input (1×28×28)
→ Conv2d(1→32, 3×3) → BatchNorm2d → ReLU
→ Conv2d(32→32, 3×3) → BatchNorm2d → ReLU
→ MaxPool2d(2×2) → Dropout(0.25)
→ Conv2d(32→64, 3×3) → BatchNorm2d → ReLU
→ Conv2d(64→64, 3×3) → BatchNorm2d → ReLU
→ MaxPool2d(2×2) → Dropout(0.25)
→ Flatten (64×7×7 = 3136)
→ Linear(3136→256) → BatchNorm1d → ReLU → Dropout(0.5)
→ Linear(256→10) [raw logits]
- **Loss:** CrossEntropyLoss (applies log_softmax internally — no softmax in model)
- **Optimizer:** Adam (lr=0.001)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Epochs:** 15
- **Batch size:** 64
- **Data:** torchvision.datasets.MNIST — auto-downloaded

## Key PyTorch concepts learned
- **`nn.Module` class pattern** — defining layers in `__init__`,
  forward pass in `forward()`. Gives complete control over computation
  flow vs Keras's declarative layer stacking.
- **Manual training loop** — `optimizer.zero_grad()` →
  `loss.backward()` → `optimizer.step()` must be called explicitly
  every batch. Forgetting `zero_grad()` accumulates gradients from
  previous batches, producing wrong weight updates.
- **`model.train()` vs `model.eval()`** — Dropout and BatchNormalization
  behave differently in training vs inference mode. PyTorch requires
  explicit switching; Keras handles this automatically inside `fit()`.
- **`torch.no_grad()`** — disables gradient computation during
  evaluation, saving memory and speeding up inference.
- **Channels-first format** — PyTorch uses (N, C, H, W) vs Keras's
  (N, H, W, C). Every reshape and display operation must account for
  this difference.
- **No softmax in model** — CrossEntropyLoss applies log_softmax
  internally. Adding softmax in the model AND in the loss applies it
  twice, producing incorrect gradients — same as Keras's `from_logits=True`.

## How to run
1. Clone the repo
```bash
   git clone https://github.com/SoheilKhdpnh/CNN-beginner-to-advance-project.git
```
2. Open in Google Colab (GPU recommended)
3. Install dependencies
```bash
   pip install torch torchvision matplotlib numpy
```
4. Open the notebook
pytorch/01-mnist-classifier/notebooks/pytorch_mnist.ipynb
## Dataset
MNIST via `torchvision.datasets.MNIST` — downloads automatically.
70,000 grayscale 28×28 images — 60,000 train / 10,000 test — 10 classes.
Normalized with MNIST mean (0.1307) and std (0.3081).
