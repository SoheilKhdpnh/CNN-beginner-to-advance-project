<h1 align="center">CNN Projects — Beginner to Advanced</h1>

<p align="center">
  A structured portfolio of Convolutional Neural Network projects built from scratch,
  progressing from basic image classification to object detection and generative models.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Keras-red?style=flat&logo=keras" />
  <img src="https://img.shields.io/badge/Status-In Progress-yellow?style=flat" />
</p>

---

## About This Repository

This repository documents my journey learning Convolutional Neural Networks,
starting from simple digit classification and working up to real-world detection
and generation tasks. Each project is self-contained with its own notebook,
results, and writeup.

---

## Projects

### Beginner

| # | Project | Description | Dataset | Accuracy | Notebook |
|---|---------|-------------|---------|----------|----------|
| 1 | MNIST Digit Classifier | Classifies handwritten digits 0–9 using a custom 2-block CNN | MNIST (70K images) | 99.1% | [Open](beginner/01-mnist-classifier/) |
| 2 | Fashion MNIST Classifier | Classifies 10 clothing categories with BatchNorm and augmentation | Fashion MNIST (70K images) | 93.2% | [Open](beginner/02-fashion-mnist/) |
| 3 | Cats vs Dogs | Binary classifier — cat or dog from real photos | Microsoft (25K images) | 96.49% | [Open](beginner/03-cats-vs-dogs/) |

### Intermediate

| # | Project | Description | Dataset | Metric | Notebook |
|---|---------|-------------|---------|--------|----------|
| 4 | Plant Disease Detector | Diagnoses plant disease from leaf photos using ResNet50 transfer learning | PlantVillage (54K images) | 99.38% | [Open](intermediate/01-plant-disease-detector/) |
| 5 | Facial Emotion Recognition | Classifies 7 facial expressions, handles severe class imbalance with weighted loss | FER-2013 (36K images) | 62.72% | [Open](intermediate/02-facial-emotion-recognition/) |
| 6 | Traffic Sign Classifier | 43-class sign recognition with MobileNetV2 + TFLite edge deployment (60% smaller, 100x faster single-image inference) | GTSRB (50K images) | 76.22% | [Open](intermediate/03-traffic-sign-classifier/) |

### Advanced

| # | Project | Description | Dataset | Metric | Notebook |
|---|---------|-------------|---------|--------|----------|
| 7 | X-Ray Pneumonia Detection | DenseNet121 transfer learning with Grad-CAM explainability, optimized for recall over accuracy | Chest X-Ray (5.8K images) | 90.54% acc / 95.38% recall | [Open](advanced/01-pneumonia-detection/) |
| 8 | Object Detection | YOLOv8 real-time detection on images + video, FPS benchmark across model sizes, FastAPI deployment | COCO (pretrained) | 89.7 FPS video / 72 FPS static | [Open](advanced/02-object-detection/) |
| 9 | Image Captioning | Coming soon | MS-COCO | — | — |

---

## Results Snapshot

| Project | Train Acc | Val Acc | Test Acc |
|---------|-----------|---------|----------|
| MNIST Classifier | 99.4% | 99.2% | 99.1% |
| Fashion MNIST | 94.1% | 93.4% | 93.2% |
| Cats vs Dogs | — | — | — |
| Traffic Sign Classifier | 96.4% | 77.3% | 76.2% |
| X-Ray Pneumonia Detection | 96.0% | 95.7% recall | 90.5% acc / 95.4% recall |
| Object Detection (YOLOv8n) | — | — | 72 FPS (static) / 89.7 FPS (video) |
---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core language |
| TensorFlow / Keras | Model building and training |
| NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| scikit-learn | Evaluation metrics |
| Jupyter Notebook | Development environment |

---

## How to Run Any Project

1. Clone the repository
```bash
   git clone https://github.com/soheilkhdpnh/CNN-projects-beginner-to-advance.git
   cd CNN-projects-beginner-to-advance
```

2. Install dependencies
```bash
   pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter
```

3. Navigate to any project and open the notebook
```bash
   cd beginner/01-mnist-classifier/notebooks
   jupyter notebook mnist_classifier.ipynb
```

---

## Repository Structure
