# ML-Ops Image Classification

This project implements an image classification pipeline using ResNet18.

## Overview

The system is designed for image classification tasks, focusing on balancing Precision and Recall.

### Threshold Optimization
Initial deployment showed high False Positives. In a medical context, we optimized the threshold to 0.9 to balance Precision and Recall, ensuring the system provides actionable alerts to doctors without causing alarm fatigue.

## Usage

### Prerequisites
- Python 3.x
- PyTorch
- torchvision

### Training
To train the model, run:
```bash
python python/train.py
```

### Evaluation
To evaluate the model, run:
```bash
python python/evaluate.py
```
