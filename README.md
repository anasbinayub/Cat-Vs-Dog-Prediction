# 🐱🐶 Cats vs Dogs Image Classification Using Deep CNN

This repository contains a **deep learning project for binary image classification** that distinguishes between **cats and dogs** using a **Convolutional Neural Network (CNN)** implemented in **PyTorch**.

The project demonstrates an **end-to-end deep learning workflow**, including dataset handling, preprocessing, model training, evaluation, and visualization. It is designed to run efficiently on **Google Colab with GPU support**.

---

## 📌 Project Overview

- **Problem Type:** Binary image classification  
- **Classes:** Cat, Dog  
- **Deep Learning Framework:** PyTorch  
- **Model Type:** Custom Convolutional Neural Network (CNN)  
- **Platform:** Google Colab (recommended)  

The goal of this project is to build a **robust CNN model** capable of accurately classifying cat and dog images from a real-world dataset.

---

## 🗂 Dataset Information

- **Dataset Name:** Cats and Dogs Classification Dataset  
- **Source:** Kaggle  
- **Dataset Link:**  
  https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset  

### 📊 Dataset Description
- Contains labeled images of **cats and dogs**
- Images vary in size and quality
- Some corrupted images exist (handled programmatically)

To ensure efficient training, the dataset is **filtered and capped at a maximum of 5,000 images per class**.

---

## ⚙️ Project Workflow

### 1. Environment Setup
- Google Drive is mounted to access dataset files
- GPU is automatically used if available
- Random seeds are fixed for reproducibility

### 2. Data Preprocessing
- Images resized for faster processing
- Normalization applied using PyTorch transforms
- Corrupted images automatically filtered out
- Dataset size balanced and limited per class

### 3. Dataset Splitting
- **80% Training data**
- **20% Testing data**
- Custom PyTorch `Subset` used for clean splitting

### 4. Data Loading
- Efficient batch loading using `DataLoader`
- Optimized with multiple workers and pinned memory

---

## 🧠 Model Architecture

A **custom deep CNN** is implemented using PyTorch:

- Convolutional layers for feature extraction
- ReLU activation functions
- Max pooling layers for dimensionality reduction
- Fully connected layers for classification
- Softmax output for class probabilities

### Training Configuration
- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Batch Size:** 64  
- **Device:** GPU (if available), otherwise CPU  

---

## 📈 Model Training & Evaluation

During training:
- Training and testing loss are recorded
- Training and testing accuracy are monitored
- Performance curves are plotted for analysis

### Evaluation Includes:
- Accuracy comparison (Train vs Test)
- Loss curves
- Visual inspection of predictions on test images

---

## 🖼 Sample Predictions Visualization

The project includes visualization of:
- Random test images
- True labels vs predicted labels
- Model confidence interpretation

This helps validate whether the model is learning meaningful visual features.

---

## ▶️ How to Run the Project

### Requirements
- Python 3.x
- Google Colab (recommended)
- Kaggle dataset downloaded and uploaded as a ZIP file

### Steps
1. Open `Cat_Vs_Dog_prediction_using_Deep_CNN.ipynb` in Google Colab
2. Upload the dataset ZIP file to Google Drive
3. Update dataset path if needed
4. Run all notebook cells sequentially
5. Model will train, evaluate, and display results

---

## 🧪 Technologies Used

- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Google Colab  

---

## 🚀 Future Improvements

- Transfer learning (ResNet, VGG, EfficientNet)
- Hyperparameter tuning
- Data augmentation
- Model deployment using Flask or FastAPI
- Web-based image upload and prediction system

---

## ⚠️ Disclaimer

This project is intended **for educational and research purposes only**.  
It is not optimized for production use without further validation and testing.

---

## 📚 Citation

If you use the dataset, please credit the original Kaggle author:

**Bhavik Jikadara**, *Cats and Dogs Classification Dataset*, Kaggle

---

## 🤝 Acknowledgements

- Kaggle community for providing the dataset
- PyTorch open-source contributors
- Google Colab for free GPU resources

---

⭐ If you find this project helpful, consider starring the repository!
