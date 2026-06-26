\# Facial Emotion Recognition



\## Overview

A Convolutional Neural Network built with TensorFlow/Keras that classifies

facial expressions into 7 emotions from the FER-2013 dataset. Trained from

scratch (no transfer learning) with class weighting to handle severe class

imbalance, particularly the underrepresented "disgust" category.



\## Results

| Metric | Score |

|--------|-------|

| Test accuracy | 62.72% |

| Classes | 7 (angry, disgust, fear, happy, neutral, sad, surprise) |

| Train images | 25,841 |

| Val images | 2,868 |

| Test images | 7,178 |



!\[Accuracy curve](results/accuracy\_curve.png)

!\[Confusion matrix](results/confusion\_matrix.png)



\## Per-class performance

| Emotion | Precision | Recall | F1-score | Support |

|---------|-----------|--------|----------|---------|

| Angry | 0.52 | 0.56 | 0.54 | 958 |

| Disgust | 0.35 | 0.76 | 0.48 | 111 |

| Fear | 0.53 | 0.32 | 0.39 | 1024 |

| Happy | 0.87 | 0.83 | 0.85 | 1774 |

| Neutral | 0.54 | 0.72 | 0.62 | 1233 |

| Sad | 0.53 | 0.41 | 0.46 | 1247 |

| Surprise | 0.69 | 0.82 | 0.75 | 831 |



\## Architecture

\- \*\*Optimizer:\*\* Adam with ReduceLROnPlateau

\- \*\*Loss:\*\* Categorical Crossentropy

\- \*\*Class weighting:\*\* sklearn's `compute\_class\_weight(balanced)` — disgust

&#x20; received a 9.39x weight vs happy's 0.57x to counter a 16x sample imbalance

\- \*\*No pretrained backbone\*\* — images are native 48×48 grayscale crops,

&#x20; too small/low-detail to benefit from ImageNet-pretrained features



\## Key learnings

\- \*\*Class weighting trades precision for recall on rare classes.\*\* Disgust's

&#x20; recall reached 0.76 (correctly catching most disgust faces) but precision

&#x20; was only 0.35 — the model over-predicts disgust as a side effect of being

&#x20; pushed to take it seriously. This is a deliberate tradeoff: missing a rare

&#x20; emotion entirely is worse than occasionally over-flagging it.

\- \*\*Fear is the hardest class\*\* (f1: 0.39) — likely confused with surprise

&#x20; and sad, since both involve overlapping facial muscle patterns (wide eyes,

&#x20; raised brows) that are subtle even for humans to distinguish at 48×48

&#x20; resolution.

\- \*\*Happy and surprise are the easiest\*\* (f1: 0.85, 0.75) — these have the

&#x20; most visually distinct, exaggerated facial cues.

\- \*\*Transfer learning isn't always the right tool.\*\* Unlike the Plant Disease

&#x20; Detector, this project intentionally skipped ImageNet-pretrained backbones

&#x20; since they're built for natural photo features, not the small-scale

&#x20; texture differences that separate facial expressions.



\## Limitations

\- The model expects a tightly-cropped face image, similar to the training

&#x20; data. A full photo with background and body will likely perform poorly —

&#x20; a production version would need a face detector (e.g. OpenCV Haar Cascade)

&#x20; to crop the face region before classification.

\- FER-2013's labels are known to be noisy/ambiguous even to humans in some

&#x20; cases, which caps the realistic accuracy ceiling for any model on this

&#x20; dataset (published benchmarks typically range 65-72%).



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

\[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)

35,887 grayscale 48×48 face images — 7 emotion classes, notably imbalanced

(disgust: 436 images vs happy: 7,215 images in the training set)

