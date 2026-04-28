# 🐱🐶 Cat vs Dog Binary Image Classification using CNN

A PyTorch-based Convolutional Neural Network for binary image classification, achieving **88.74% accuracy** on the Cats vs Dogs dataset with comprehensive evaluation metrics.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Problem Statement

Image classification is one of the most widely studied problems in computer vision. One common benchmark problem is distinguishing between images of cats and dogs, which requires the model to learn complex visual patterns such as shapes, textures, and edges.

The goal of this project is to develop a **Convolutional Neural Network (CNN)** to perform binary image classification using the **Dogs vs Cats dataset**. The dataset contains thousands of labeled images of cats and dogs with varying backgrounds, lighting conditions, and poses.

The CNN model automatically learns relevant visual features from raw images and classifies each image into one of two categories: **cat or dog**.

The project includes dataset preprocessing, CNN model design, training and validation, and evaluation using metrics such as **accuracy, precision, recall, and confusion matrix**.

## 🎯 Key Features

✅ Custom CNN architecture with 3 convolutional layers  
✅ BatchNormalization for stable training  
✅ Dropout (0.5) for regularization  
✅ Data augmentation (Random Horizontal Flip)  
✅ Adam optimizer with weight decay (L2 regularization)  
✅ Comprehensive evaluation metrics  
✅ Pre-trained model weights included

## 🏗️ Model Architecture

Input (3×64×64) → Conv2D(32) → BatchNorm → ReLU → MaxPool(2×2) → Conv2D(64) → BatchNorm → ReLU → MaxPool(2×2) → Conv2D(128) → BatchNorm → ReLU → MaxPool(2×2) → Flatten (8×8×128 = 8192) → FC(256) → ReLU → Dropout(0.5) → FC(2) → Output [Cat/Dog]

**Parameters:** Input Size: 64×64×3 (RGB images) | Total Trainable Parameters: ~2.1M | Optimizer: Adam (lr=0.0005, weight_decay=1e-4) | Loss Function: CrossEntropyLoss

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | **88.74%** |
| **Precision** | **91.56%** |
| **Recall** | **85.48%** |
| **Final Train Loss** | 0.2781 |
| **Final Val Loss** | 0.2804 |

**Confusion Matrix:** Actual Cat [2288 198] | Actual Dog [365 2149]

## 🛠️ Tech Stack

**Framework:** PyTorch 2.0+ | **Libraries:** torchvision, NumPy, scikit-learn, Matplotlib, Pillow | **Language:** Python 3.13 | **Model Type:** Convolutional Neural Network (CNN)

## 📁 Project Structure

CatDog-CNN-Classifier/ contains BinaryImageClassificationusingCNN.ipynb (Main implementation), cat_dog_model.pth (Trained model weights), requirements.txt (Python dependencies), and README.md (Project documentation)

## 🚀 Installation & Usage

**1. Clone the Repository**
```bash
git clone https://github.com/paramramit305-a11y/CatDog-CNN-Classifier.git
cd CatDog-CNN-Classifier
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Download Dataset:** This project uses the Kaggle Cats and Dogs Dataset. Download from [Kaggle - Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data) and place the extracted PetImages folder in the project root before running the notebook.

**4. Run the Notebook**
```bash
jupyter notebook BinaryImageClassificationusingCNN.ipynb
```

**5. Load Pre-trained Model**
```python
import torch
from model import CNN

model = CNN()
model.load_state_dict(torch.load('cat_dog_model.pth'))
model.eval()
```

## 🔑 Key Techniques Used

Data Augmentation (Random horizontal flips) | Batch Normalization (Stabilizes training) | Dropout Regularization (p=0.5 reduces overfitting) | L2 Regularization (Weight decay in Adam optimizer) | Train-Validation Split (80-20 split)

## 📈 Training Details

**Epochs:** 10 | **Batch Size:** 32 | **Optimizer:** Adam | **Learning Rate:** 0.0005 | **Weight Decay:** 1e-4 | **Image Size:** 64×64 | **Normalization:** Mean & Std = (0.5, 0.5, 0.5)

## 💡 Future Improvements

- Experiment with deeper architectures (ResNet, VGG)
- Implement learning rate scheduling
- Add more data augmentation techniques
- Deploy model using Flask/Streamlit
- Achieve 90%+ accuracy with hyperparameter tuning

## 🔗 Connect With Me

**GitHub:** [paramramit305-a11y](https://github.com/paramramit305-a11y) | **LinkedIn:** [Parmar Amit](https://www.linkedin.com/in/parmar-amit-97941a377)

## 👨‍💻 Author

**Parmar Amit** - *Aspiring AI Engineer | B.Sc. IT (AI/ML) Student*

## 📜 License

This project is licensed under the MIT License.

---

⭐ **If you find this project helpful, please give it a star!**
