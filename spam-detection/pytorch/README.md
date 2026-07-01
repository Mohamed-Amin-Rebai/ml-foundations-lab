# 📩 Spam Detection using PyTorch

This project implements a spam detection system using a custom neural network built with PyTorch.

Unlike the Scikit-Learn version, this implementation learns text representations using an embedding layer and trains a neural network to classify messages as:

- 🚨 Spam
- ✅ Ham (Not Spam)

---

# 🎯 Goal

Build a text classification system using PyTorch and understand the complete deep learning workflow for NLP.

---

# ⚙️ Technologies Used

- Python
- Pandas
- PyTorch
- Torch DataLoader
- Neural Networks
- Embeddings
- Joblib

---

# 📂 Project Structure

```text
spam-detection/
└── pytorch/
    ├── dataset/
    │   └── spam.csv
    │
    ├── model/
    │   ├── spam_model.pth
    │   └── vocab.pkl
    │
    ├── dataset_loader.py
    ├── train.py
    ├── main.py
    └── README.md
```

---

# 📦 Dataset

This project uses the SMS Spam Collection Dataset.

Each message belongs to one of two classes:

| Label | Meaning |
|---------|---------|
| ham | Legitimate message |
| spam | Unwanted promotional message |

Example:

```text
ham, Hey bro are we still meeting tomorrow?
spam, Congratulations! You have won a FREE prize. Call now to claim.
```

---

# 🧠 Deep Learning Pipeline

```text
SMS Dataset
      ↓
Tokenization
      ↓
Vocabulary Creation
      ↓
Text → Word IDs
      ↓
Embedding Layer
      ↓
Neural Network
      ↓
Training
      ↓
Model Saving
      ↓
Prediction
```

---

# 🔍 Text Processing

Messages are converted into tokens:

```text
"free money now"

↓
["free", "money", "now"]
```

A vocabulary is built:

```python
{
    "<PAD>": 0,
    "<UNK>": 1,
    "free": 2,
    "money": 3,
    ...
}
```

Then tokens are converted into numerical IDs:

```text
["free", "money"]

↓

[2, 3]
```

---

# 🤖 Model Architecture

```text
Input IDs
     ↓
Embedding Layer
     ↓
Average Pooling
     ↓
Linear Layer
     ↓
ReLU
     ↓
Linear Layer
     ↓
Spam / Ham
```

The network learns patterns from text and predicts whether a message is spam or legitimate.

---

# 📊 Results

Training Results:

```text
Accuracy: 97.22%
```

The model achieved performance comparable to the Scikit-Learn implementation while being built entirely with PyTorch.

---

# ▶️ Training

Run:

```bash
python train.py
```

This will:

- Load the dataset
- Build the vocabulary
- Create DataLoaders
- Train the neural network
- Evaluate performance
- Save the trained model

Generated files:

```text
model/
├── spam_model.pth
└── vocab.pkl
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
Congratulations! You have won a FREE prize. Call now to claim.
```

Output:

```text
🚨 SPAM
Confidence: 98.45%
```

Example:

Input:

```text
Hey bro what time are we meeting tomorrow?
```

Output:

```text
✅ HAM
Confidence: 99.94%
```

---

# 💾 Saved Artifacts

The trained model is stored as:

```text
spam_model.pth
```

The vocabulary used during training is stored as:

```text
vocab.pkl
```

These files are loaded during inference.

---

# 🧠 Key Learning Outcomes

Through this project, I learned:

- Tokenization
- Vocabulary Creation
- Text Numerical Encoding
- Embedding Layers
- PyTorch Datasets
- PyTorch DataLoaders
- Neural Network Training
- Cross Entropy Loss
- Adam Optimizer
- Model Persistence
- Deep Learning Inference

---

# 🚀 Future Improvements

- LSTM Architecture
- GRU Architecture
- Attention Mechanism
- BERT-Based Spam Detection
- FastAPI Deployment
- Streamlit Web Interface
- Model Explainability

---

# 🔗 Related Projects

- Sentiment Analysis (Rule-Based NLP)
- Sentiment Analysis (Pretrained BERT)
- Spam Detection (Scikit-Learn)
- Image Classification (Cats vs Dogs)

---

# 👨‍💻 Author

**Mohamed Amine REBAI**

*Software Engineering Student | AI Enthusiast*