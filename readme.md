# Vehicle Type Classification System

## Objective
Classify vehicle images into:
- Car
- Truck
- Bike
- Bus
- Ambulance

## Dataset
- Source: (Mention your dataset source here)
- Classes: 5
- Images: (mention count per class)
- Resolution: Resized to 224x224

## Preprocessing
- Image resizing
- Normalization (0–1)
- Data augmentation:
  - Rotation
  - Zoom
  - Flip

## Model
- Convolutional Neural Network (CNN)
- 3 convolution layers
- Softmax output layer

## Evaluation
- Accuracy used as primary metric

## Decision Layer
- ≥ 0.85 → High Confidence
- 0.65 – 0.85 → Needs Review
- < 0.65 → Uncertain

## How to Run

### Train Model
```bash
python app.py
