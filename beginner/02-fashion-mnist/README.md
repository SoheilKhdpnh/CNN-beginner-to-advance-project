\# Fashion MNIST Classifier



\## Overview

A CNN built with TensorFlow/Keras that classifies 10 categories of clothing

from the Fashion MNIST dataset. Uses BatchNormalization, Dropout, and data

augmentation to tackle a significantly harder classification task than MNIST.



\## Results

| Metric | Score |

|--accuracy-|--92.93--|

| Test accuracy | 92.93% |

| Parameters | \~400K |

| Training time | \~15 min (CPU) |



!\[Sample images](results/sample\_images.png)

!\[Accuracy curve](results/accuracy\_curve.png)

!\[Confusion matrix](results/confusion\_matrix.png)



\## Architecture



\- \*\*Optimizer:\*\* Adam with ReduceLROnPlateau

\- \*\*Loss:\*\* Categorical Crossentropy

\- \*\*Epochs:\*\* 30 (early stopping)

\- \*\*Augmentation:\*\* horizontal flip, rotation, zoom, shift



\## Key Learnings

\- Fashion MNIST is meaningfully harder than MNIST — same architecture only gets \~89%

\- BatchNormalization after every Conv layer stabilizes training noticeably

\- The confusion matrix shows Shirt vs T-shirt/top are the hardest to separate

\- Data augmentation closed the train/val accuracy gap significantly



\## How to Run

1\. Clone the repo

```bash

&#x20;  git clone https://github.com/YOUR\_USERNAME/CNN-projects-beginner-to-advance.git

```

2\. Install dependencies

```bash

&#x20;  pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter

```

3\. Open the notebook

```bash

&#x20;  cd beginner/02-fashion-mnist/notebooks

&#x20;  jupyter notebook fashion\_mnist\_classifier.ipynb

```



\## Dataset

\[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) via `keras.datasets.fashion\_mnist`

70,000 grayscale 28×28 images — 60,000 train, 10,000 test — 10 clothing classes

