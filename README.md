# 🧠 ML Foundations Lab

> A collection of fundamental Machine Learning and Deep Learning projects designed to demonstrate core concepts in:
>
> - Natural Language Processing (NLP)
> - Text Classification
> - Computer Vision
>
> This repository follows a progressive learning path:
>
> **Classical Machine Learning → Deep Learning**
>
> showcasing how different techniques solve similar real-world problems.

---

## 🚀 Projects Overview

---

## 📌 1. Sentiment Analysis

### 🔹 Simple Approaches

- ✅ TextBlob
- ✅ VADER

#### 🎯 Goal

Classify text sentiment *(Positive / Negative / Neutral)* and compare rule-based approaches.

#### 📚 Learning Outcomes

- Lexicon-based NLP
- Sentiment scoring
- Handling informal text using VADER

---

### 🔹 Deep Learning (BERT)

Using a **Transformer-based model (BERT)** via HuggingFace.

#### 🎯 Goal

Achieve more accurate sentiment classification.

#### 📚 Learning Outcomes

- Tokenization
- Pretrained models
- Transfer Learning in NLP

---

## 📌 2. Spam Detection (Text Classification)

### 🔹 Scikit-learn Approach

#### Models

- ✅ Naive Bayes
- ✅ Logistic Regression *(optional)*

#### Pipeline

- Text preprocessing
- Feature extraction *(Bag of Words / TF-IDF)*
- Model training

#### 📚 Learning Outcomes

- Classical ML pipeline
- Feature engineering for NLP

---

### 🔹 PyTorch Approach

Using a custom neural network with:

- Embedding Layer
- Average Pooling
- Feedforward Neural Network

#### 🎯 Goal

Compare classical Machine Learning with Deep Learning approaches for text classification.

#### 📚 Learning Outcomes

- Tokenization
- Vocabulary Creation
- Text Numerical Encoding
- Embedding Layers
- Neural Network Training

---

## 📌 3. Image Classification (Cats vs Dogs)

### Framework

**PyTorch + Torchvision**

### Approaches

- ✅ Transfer Learning (ResNet18)

#### 🎯 Goal

Classify images into categories:

- 🐱 Cat
- 🐶 Dog

#### 📚 Learning Outcomes

- CNN fundamentals
- Image preprocessing
- Transfer Learning *(VERY IMPORTANT)*

---

## 📌 4. Machine Learning Techniques

A collection of classical Machine Learning projects covering regression, classification, clustering, and dimensionality reduction.

### Topics Covered

- ✅ Simple Linear Regression
- ✅ Multiple Linear Regression
- ✅ Diabetes Classification
- ✅ Iris Classification
- ✅ Brain Cancer Classification
- ✅ K-Means Clustering

#### 📚 Learning Outcomes

- Data Cleaning
- Missing Value Handling
- Feature Engineering
- Feature Scaling
- Classification Algorithms
- Regression Analysis
- Clustering
- PCA (Dimensionality Reduction)
- Model Evaluation

---

## 📂 Project Structure

```text
.
├── sentiment-analysis/
│   ├── pre-trained-bert-model/
│   └── rule-based-nlp/
│
├── spam-detection/
│   ├── sklearn/
│   └── pytorch/
│
├── image-classification/
│   └── pytorch/
│
├── learning-techniques/
│   ├── linear_regression.ipynb
│   ├── classification.ipynb
│   ├── classification_iris.ipynb
│   ├── brain_cancer.ipynb
│   └── clustering.ipynb
│
│
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Mohamed-Amin-Rebai/ml-foundations-lab.git

cd ml-foundations-lab

pip install -r requirements.txt
```

---

## 📦 Requirements

Main libraries:

```text
torch
torchvision
transformers
scikit-learn
pandas
numpy
matplotlib
textblob
vaderSentiment
```

---

## ▶️ Usage

Each module is independent.

Example:

```bash
cd sentiment-analysis/rule-based-nlp

python main.py
```

---

## 📊 Results

| Project | Accuracy |
| :------- | -------: |
| Spam Detection (Scikit-Learn) | **96.86%** |
| Spam Detection (PyTorch) | **97.22%** |
| Cats vs Dogs (ResNet18) | **95.72%** |

---

## 🛠️ Techniques Covered

| Category | Techniques |
|-----------|-----------|
| Regression | Linear Regression, Multiple Linear Regression |
| Classification | Logistic Regression, KNN, SVM, Decision Trees, Random Forests, Neural Networks |
| Clustering | K-Means |
| NLP | TextBlob, VADER, TF-IDF, BERT |
| Computer Vision | ResNet18 Transfer Learning |
| Deep Learning | PyTorch Neural Networks |

---


## 📚 Key Learning Outcomes

- ✔️ Understand the difference between:
  - Rule-based NLP
  - Machine Learning
  - Deep Learning
- ✔️ Learn text preprocessing and vectorization
- ✔️ Build and train neural networks using PyTorch
- ✔️ Apply transfer learning for NLP and Computer Vision
- ✔️ Compare model performance across approaches
- ✔️ Apply Regression, Classification, and Clustering techniques
- ✔️ Understand supervised and unsupervised learning workflows

---

## 🧠 Why this project?

This repository is designed to:

- Build strong ML foundations
- Demonstrate practical understanding of ML pipelines
- Showcase real-world AI problem solving

---

## 🚀 Future Improvements

- 📈 Add evaluation dashboard
- ⚡ FastAPI API
- 🤖 Model serving
- 🌐 Deploy mini apps

---

## 👨‍💻 Author

**Mohamed Amine REBAI**

*Software Engineering Student | AI Enthusiast*