# 🧠 Perceptron vs ANN vs CNN — MNIST Digit Classification

A comparative deep learning study implementing three neural network architectures — **Perceptron**, **Artificial Neural Network (ANN)**, and **Convolutional Neural Network (CNN)** — on the classic **MNIST handwritten digit dataset**. Built using TensorFlow/Keras, this notebook benchmarks model complexity vs accuracy across all three paradigms.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Training Configuration](#training-configuration)
- [Model Comparison](#model-comparison)
- [How to Run](#how-to-run)
- [Key Concepts](#key-concepts)
- [Results Summary](#results-summary)

---

## 📖 Overview

This notebook demonstrates how increasing neural network complexity leads to better image classification performance. Starting from the simplest linear model (Perceptron) and progressing through a fully-connected ANN to a spatial-feature-aware CNN, the project provides an end-to-end understanding of deep learning fundamentals.

---

## 📊 Dataset

| Property         | Details                                      |
|------------------|----------------------------------------------|
| **Source**       | `train.csv` (MNIST format)                   |
| **Task**         | Multi-class Image Classification (10 classes)|
| **Classes**      | Digits 0–9                                   |
| **Image Size**   | 28 × 28 pixels (grayscale)                   |
| **Features**     | 784 pixel values per image                   |
| **Target Column**| `label`                                      |
| **Normalization**| Pixel values scaled to `[0.0, 1.0]` (÷ 255) |
| **Label Format** | One-Hot Encoded via `to_categorical`         |
| **Train/Test**   | 75% / 25% split (default `train_test_split`) |

---

## 🛠️ Tech Stack

| Category          | Libraries / Tools                                      |
|-------------------|--------------------------------------------------------|
| **Language**      | Python 3.x                                             |
| **Data Handling** | NumPy, Pandas                                          |
| **Visualization** | Matplotlib, Seaborn                                    |
| **Preprocessing** | Scikit-learn (`LabelEncoder`, `StandardScaler`, `train_test_split`) |
| **Deep Learning** | TensorFlow, Keras (`Sequential`, `Dense`, `Conv2D`, etc.) |
| **Metrics**       | `accuracy_score`, `classification_report`, `confusion_matrix` |

---

## 📁 Project Structure

```
perceptron_ANN_CNN/
│
├── perceptron_ANN_CNN.ipynb    # Main notebook with all three models
├── train.csv                   # MNIST training dataset (CSV format)
└── README.md                   # Project documentation
```

---

## 🏗️ Model Architectures

### 1. 🔵 Perceptron (Single-Layer)

The simplest neural network — a single `Dense` layer with `softmax` activation. No hidden layers; purely linear decision boundaries.

```
Input (784)  →  Flatten (28×28)  →  Dense(10, softmax)
```

| Layer     | Type    | Units | Activation |
|-----------|---------|-------|------------|
| Input     | Flatten | 784   | —          |
| Output    | Dense   | 10    | Softmax    |

- **Optimizer:** SGD (Stochastic Gradient Descent)
- **Loss:** Categorical Crossentropy

---

### 2. 🟡 ANN (Multi-Layer Perceptron)

A fully-connected network with two hidden layers using ReLU activations, enabling non-linear feature learning.

```
Input (784)  →  Flatten  →  Dense(128, ReLU)  →  Dense(64, ReLU)  →  Dense(10, Softmax)
```

| Layer    | Type    | Units | Activation |
|----------|---------|-------|------------|
| Input    | Flatten | 784   | —          |
| Hidden 1 | Dense   | 128   | ReLU       |
| Hidden 2 | Dense   | 64    | ReLU       |
| Output   | Dense   | 10    | Softmax    |

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy

---

### 3. 🟢 CNN (Convolutional Neural Network)

the most powerful of the three — uses convolutional layers to extract spatial features, pooling to reduce dimensionality, and dropout to prevent overfitting.

```
Input (28×28×1)
    → Conv2D(32, 3×3, ReLU)
    → MaxPooling2D(2×2)
    → Conv2D(64, 3×3, ReLU)
    → MaxPooling2D(2×2)
    → Flatten
    → Dense(128, ReLU)
    → Dropout(0.5)
    → Dense(10, Softmax)
```

| Layer        | Type          | Filters/Units | Kernel  | Activation |
|--------------|---------------|---------------|---------|------------|
| Conv Block 1 | Conv2D        | 32            | 3 × 3   | ReLU       |
|              | MaxPooling2D  | —             | 2 × 2   | —          |
| Conv Block 2 | Conv2D        | 64            | 3 × 3   | ReLU       |
|              | MaxPooling2D  | —             | 2 × 2   | —          |
| FC Layer     | Dense         | 128           | —       | ReLU       |
| Regularizer  | Dropout       | 50%           | —       | —          |
| Output       | Dense         | 10            | —       | Softmax    |

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Input Shape:** `(28, 28, 1)` — grayscale channel

---

## ⚙️ Training Configuration

| Parameter       | Perceptron | ANN    | CNN    |
|-----------------|------------|--------|--------|
| **Epochs**      | 5          | 5      | 5      |
| **Batch Size**  | 32         | 32     | 32     |
| **Optimizer**   | SGD        | Adam   | Adam   |
| **Loss**        | Categorical Crossentropy | Categorical Crossentropy | Categorical Crossentropy |
| **Validation**  | ✅ Yes     | ✅ Yes | ✅ Yes |
| **Input Shape** | (28, 28)   | (28, 28) | (28, 28, 1) |

---

## 📈 Model Comparison

| Model           | Architecture Type     | Parameters | Expected Accuracy |
|-----------------|-----------------------|------------|--------------------|
| **Perceptron**  | Single-Layer Linear   | ~7,850     | ~90–92%            |
| **ANN**         | Multi-Layer FC        | ~109,386   | ~96–97%            |
| **CNN**         | Conv + Pooling + FC   | ~93,322    | ~98–99%            |

> ⚠️ Actual accuracy values depend on your random seed and train/test split. Run `perceptron.evaluate()`, `ann.evaluate()`, `cnn.evaluate()` to get final test accuracy.

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/perceptron-ANN-CNN.git
cd perceptron-ANN-CNN
```

### 2. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

### 3. Add the Dataset

Place your `train.csv` (MNIST CSV format) in the project root. You can download it from [Kaggle – Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data).

### 4. Launch the Notebook

```bash
jupyter notebook perceptron_ANN_CNN.ipynb
```

### 5. Run All Cells

Execute cells sequentially — preprocessing → Perceptron → ANN → CNN → comparison.

---

## 💡 Key Concepts

| Concept               | What It Does                                                                 |
|-----------------------|------------------------------------------------------------------------------|
| `Flatten`             | Converts 2D image arrays (28×28) into 1D vectors (784) for Dense layers     |
| `Dense`               | Fully-connected layer — each neuron connects to all neurons in the next layer|
| `ReLU`                | Activation — outputs `max(0, x)`, enables non-linear learning                |
| `Softmax`             | Converts output logits to class probabilities (sums to 1)                    |
| `Conv2D`              | Extracts local spatial features (edges, textures) using learnable filters    |
| `MaxPooling2D`        | Reduces spatial dimensions; retains dominant features, reduces computation   |
| `Dropout(0.5)`        | Randomly drops 50% of neurons during training to prevent overfitting         |
| `to_categorical`      | Converts integer labels to one-hot encoded vectors (e.g. `3` → `[0,0,0,1,...,0]`) |
| `Adam`                | Adaptive learning rate optimizer — faster convergence than plain SGD         |
| `SGD`                 | Stochastic Gradient Descent — simpler optimizer, used for Perceptron baseline|
| `categorical_crossentropy` | Loss function for multi-class classification with one-hot labels        |

---

## 📋 Results Summary

After training all three models for 5 epochs on the MNIST dataset, the expected accuracy ranking is:

```
CNN  >  ANN  >  Perceptron
```

- The **Perceptron** fails to capture non-linear patterns — linear decision boundary only.
- The **ANN** improves significantly with hidden layers and ReLU activations.
- The **CNN** achieves the best accuracy by exploiting spatial structure in images through convolutional feature extraction.

---

## 👨‍💻 Author

**Aayush**
B.Tech Information Technology | Data Science & ML Enthusiast
[GitHub](https://github.com/Aayush20art) · [LinkedIn](www.linkedin.com/in/aayush-sharma-b108a93b0)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
