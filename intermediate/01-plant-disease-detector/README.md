\# Plant Disease Detector



\## Overview

A multi-class image classifier built with TensorFlow/Keras that identifies

plant species and diagnoses disease (or confirms health) from a leaf photo.

Uses transfer learning with ResNet50 pretrained on ImageNet, fine-tuned on

the PlantVillage dataset across 38 classes.



\## Results

| Metric  | Score |

| accuracy|-------|

| Test accuracy | 99.45% |

| Classes | 38 (plant species × disease/healthy) |

| Base model | ResNet50 (ImageNet weights) |

| Input size | 224×224×3 |



!\[Accuracy curve](results/accuracy\_curve.png)

!\[Confusion matrix](results/confusion\_matrix.png)



\## Architecture

\*\*Two-phase training:\*\*

1\. \*\*Frozen base\*\* — only the classification head trained, 10 epochs

2\. \*\*Fine-tuning\*\* — last 30 layers of ResNet50 unfrozen, trained with a much

&#x20;  lower learning rate (1e-5), 20 epochs



\- \*\*Optimizer:\*\* Adam (lr=0.001 frozen phase → 1e-5 fine-tune phase)

\- \*\*Loss:\*\* Categorical Crossentropy

\- \*\*Augmentation:\*\* rotation, horizontal flip, zoom, width/height shift



\## Key learnings

\- Transfer learning with a frozen ImageNet backbone reached a strong baseline

&#x20; almost immediately compared to training a CNN from scratch

\- Fine-tuning the last 30 layers with a low learning rate pushed accuracy

&#x20; noticeably higher than the frozen-base baseline

\- Using a very low learning rate during fine-tuning is essential — a normal

&#x20; learning rate destroys the pretrained weights almost instantly

\- With 38 well-separated classes (different plants + diseases), the model

&#x20; rarely confuses unrelated categories; most residual errors are between

&#x20; visually similar disease stages of the same plant



\## How to run

1\. Clone the repo

```bash

&#x20;  git clone https://github.com/SoheilKhdpnh/CNN-beginner-to-advance-project.git

```

2\. Open in Kaggle Notebooks or Google Colab (GPU recommended)

3\. Install dependencies

```bash

&#x20;  pip install tensorflow numpy matplotlib seaborn scikit-learn gradio

```

4\. Open the notebook

\## Dataset

\[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

\~54,000 color leaf images — 38 classes covering multiple plant species and

their disease states (or healthy)

