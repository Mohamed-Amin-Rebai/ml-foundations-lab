# 🐱🐶 Cats vs Dogs Image Classification using ResNet18

This project implements an image classification system using PyTorch and Transfer Learning.

A pretrained ResNet18 model is fine-tuned to classify images into:

- 🐱 Cat
- 🐶 Dog

---

# 🎯 Goal

Build an image classification model capable of identifying whether an image contains a cat or a dog.

This project focuses on Transfer Learning, a widely used Computer Vision technique that leverages pretrained models to achieve high performance with less training time.

---

# ⚙️ Technologies Used

- Python
- PyTorch
- Torchvision
- Pillow (PIL)
- ResNet18
- Transfer Learning

---

# 📂 Project Structure

```text
image-classification/
└── pytorch/
    ├── dataset/
    │   └── PetImages/
    │       ├── Cat/
    │       └── Dog/
    │
    ├── model/
    │   └── cats_dogs_model.pth
    │
    ├── train.py
    ├── main.py
    └── README.md
```

---

# 📦 Dataset

This project uses Microsoft's Cats vs Dogs dataset.

Dataset Structure:

```text
PetImages/
├── Cat/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
│
└── Dog/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

Total Images:

```text
24,998
```

Some corrupted images were removed before training.

---

# 🧠 Computer Vision Pipeline

```text
Image
   ↓
Resize (224x224)
   ↓
Tensor Conversion
   ↓
ResNet18
   ↓
Fine-Tuned Classification Layer
   ↓
Cat / Dog
```

---

# 🔄 What is Transfer Learning?

Instead of training a neural network from scratch, we use a model already trained on millions of images.

Benefits:

- Faster training
- Better accuracy
- Less data required
- Industry-standard approach

---

# 🤖 Model Architecture

Base Model:

```python
torchvision.models.resnet18()
```

Modified Final Layer:

```python
model.fc = nn.Linear(
    model.fc.in_features,
    2
)
```

Output Classes:

```text
0 → Cat
1 → Dog
```

---

# 🏋️ Training

Loss Function:

```python
CrossEntropyLoss()
```

Optimizer:

```python
Adam
```

Epochs:

```text
3
```

---

# 📊 Results

Training Results:

```text
Accuracy: 95.72%
```

This demonstrates the effectiveness of Transfer Learning using pretrained CNN architectures.

---

# ▶️ Training

Run:

```bash
python train.py
```

This will:

- Load the dataset
- Apply image transformations
- Train ResNet18
- Evaluate accuracy
- Save the trained model

Generated File:

```text
model/cats_dogs_model.pth
```

---

# ▶️ Prediction

Run:

```bash
python main.py
```

Example:

Input:

```text
test/cat.jpg
```

Output:

```text
🐱 Cat
Confidence: 99.12%
```

Example:

Input:

```text
test/dog.jpg
```

Output:

```text
🐶 Dog
Confidence: 98.54%
```

---

# 🧠 Key Learning Outcomes

Through this project, I learned:

- Computer Vision fundamentals
- Image preprocessing
- DataLoaders
- Transfer Learning
- ResNet18 architecture
- Fine-tuning pretrained models
- CNN-based classification
- Model persistence
- Image inference pipeline

---

# 🚀 Future Improvements

- Data Augmentation
- ResNet50
- EfficientNet
- MobileNet
- Multi-Class Animal Classification
- FastAPI Deployment
- Streamlit Web Interface
- Real-Time Webcam Classification

---

# 🔗 Related Projects

- Sentiment Analysis (Rule-Based NLP)
- Sentiment Analysis (Pretrained BERT)
- Spam Detection (Scikit-Learn)
- Spam Detection (PyTorch)

---

# 👨‍💻 Author

**Mohamed Amine REBAI**

*Software Engineering Student | AI Enthusiast*