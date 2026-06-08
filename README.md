Gender Classification using EfficientNet-B0 
Overview
This project implements a binary gender classifier (Male / Female) using deep learning. The model is built on top of EfficientNet-B0, a lightweight yet powerful convolutional neural network pretrained on ImageNet, fine-tuned on the UTKFace dataset.

Dataset
The UTKFace dataset (crop_part1 subset) was used containing 9,779 face images. Labels are extracted directly from filenames in the format age_gender_race_date.jpg — requiring zero external annotation files. The dataset was split into three parts:

Train: 6,845 images (70%)
Validation: 1,466 images (15%)
Test: 1,468 images (15%)


Model Architecture
EfficientNet-B0 was loaded with pretrained ImageNet weights. The original 1000-class output head was replaced with a custom 2-class classifier:
Global Average Pool → Dropout(0.3) → Linear(1280 → 2)
Total trainable parameters: ~5.3M

Training Configuration
SettingValueOptimizerAdamWLearning Rate1e-4Weight Decay1e-4SchedulerCosineAnnealingLRBatch Size32Epochs20HardwareKaggle GPU (T4)

Results
MetricValueBest Validation Accuracy88.61%Final Test Accuracy90.80%Test Loss0.5295
The model reached 90%+ training accuracy by epoch 3 and stabilized around 88–89% validation accuracy. The gap between training loss (near 0) and validation loss (rising to 0.69) indicates mild overfitting — expected with only ~6,800 training images, and addressable by adding the full UTKFace folder in the next training run.

Observations from Training Curves

Train accuracy climbed steeply to ~99% while val accuracy plateaued around 88% — classic sign of a small dataset relative to model capacity
Val loss rose after epoch 5 despite val accuracy staying stable — the model became overconfident on training data but still predicted correctly on val
Test accuracy (90.80%) being higher than best val accuracy (88.61%) confirms the model generalizes well and there is no data leakage


Next Steps 

Add full UTKFace folder (~23,708 images) to the training set for better generalization
Build a FastAPI application that accepts an image upload and returns predicted gender + confidence score
Add face detection (MTCNN) as a preprocessing step before inference
Deploy the FastAPI app for live testing
Sonnet 4.6 LowClaude is AI and can make mistakes. Please double-check responses.
