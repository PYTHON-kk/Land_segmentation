# Land Segmentation using Knowledge distillation model

A deep learning-based semantic segmentation project for analyzing aerial imagery and extracting meaningful land features such as buildings, roads, vegetation, and water bodies.
---

##  Overview

This project performs **semantic segmentation** on aerial images, where each pixel is classified into a specific land category. It helps in understanding satellite imagery for real-world applications like urban planning, agriculture, and environmental monitoring.

---

##  Dataset

Dataset used:  
https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery

### Dataset Features:
- High-resolution aerial images  
- Pixel-wise labeled masks  
- Multiple land classes (buildings, roads, vegetation, etc.)

---

## Tech Stack
- Python  
- PyTorch / TensorFlow  
- NumPy  
- OpenCV  
- Matplotlib  
- Scikit-learn

# Installation & Setup
1) Clone the repository
git clone https://github.com/PYTHON-kk/Land_segmentation.git
cd Land_segmentation
2) Install dependencies
   ```pip install -r requirements.txt```
4) Download dataset
Download dataset from Kaggle
Extract and place it inside the dataset/ folder

---

# How to Run
Training
```python train.py```

Testing / Inference
```python predict.py```

# Results
Model predicts segmentation masks for aerial images
Outputs include:
Original image
Ground truth mask
Predicted mask
