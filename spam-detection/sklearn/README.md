# 📩 Spam Detection using Scikit-Learn

This project implements a machine learning-based spam detection system using Scikit-Learn.

The model is trained on the SMS Spam Collection dataset and classifies messages as:

- 🚨 Spam
- ✅ Ham (Not Spam)

---

# 🎯 Goal

Build a text classification system capable of identifying spam messages with high accuracy.

---

# ⚙️ Technologies Used

- Python
- Pandas
- Scikit-Learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Joblib

---

# 📂 Project Structure

```text
spam-detection/
└── sklearn/
    ├── dataset/
    │   └── spam.csv
    │
    ├── model/
    │   ├── spam_model.pkl
    │   └── vectorizer.pkl
    │
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
spam, WIN a FREE iPhone now!!!
```

---

# 🧠 Machine Learning Pipeline

```text
SMS Dataset
      ↓
Text Preprocessing
      ↓
TF-IDF Vectorization
      ↓
Multinomial Naive Bayes
      ↓
Model Evaluation
      ↓
Save Model
      ↓
Prediction
```

---

# 🔍 Feature Extraction

Messages are converted into numerical vectors using:

```python
TfidfVectorizer()
```

TF-IDF helps identify words that are important for spam classification.

Examples:

```text
free
winner
claim
prize
urgent
```

These words typically receive higher importance scores in spam messages.

---

# 🤖 Model

The classifier used is:

```python
MultinomialNB()
```

Why?

- Fast to train
- Efficient on text data
- Common baseline for spam detection
- Strong performance with TF-IDF features

---

# 📊 Results

Training Results:

```text
Accuracy: 96.86%
```

Classification Report:

```text
Ham Precision: 96%
Spam Precision: 100%

Overall Accuracy: 97%
```

---

# ▶️ Training

Run:

```bash
python train.py
```

This will:

- Load the dataset
- Vectorize messages
- Train the model
- Evaluate performance
- Save the trained artifacts

Generated files:

```text
model/
├── spam_model.pkl
└── vectorizer.pkl
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
Congratulations!
You won a FREE iPhone!
Click here now!
```

Output:

```text
🚨 SPAM
```

Example:

Input:

```text
Hey bro, are we still meeting tomorrow?
```

Output:

```text
✅ HAM
```

---

# 🧠 Key Learning Outcomes

Through this project, I learned:

- Text Classification
- NLP preprocessing fundamentals
- Feature extraction using TF-IDF
- Train/Test splitting
- Naive Bayes classification
- Model evaluation metrics
- Model persistence using Joblib
- Real-world inference workflow

---

# 🚀 Future Improvements

- Logistic Regression classifier
- Random Forest classifier
- PyTorch Neural Network version
- BERT-based spam detection
- FastAPI deployment
- Streamlit Web Interface

---

# 🔗 Related Projects

- Sentiment Analysis (Rule-Based NLP)
- Sentiment Analysis (Pretrained BERT)
- Fine-Tuned BERT Sentiment Analysis
- Image Classification (Cats vs Dogs)

---

# 👨‍💻 Author

**Mohamed Amine REBAI**

Software Engineering Student | AI Enthusiast