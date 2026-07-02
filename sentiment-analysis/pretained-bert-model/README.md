# 🧠 Pretrained BERT Sentiment Analysis

> This module demonstrates **sentiment analysis using a pretrained Transformer model (BERT variant)** via Hugging Face.
>
> Unlike rule-based methods, this approach leverages **deep learning and contextual understanding** to classify text sentiment.

---

## 🎯 Goal

Classify input text into:

- ✅ Positive
- ❌ Negative

using a pretrained model without additional training.

---

## ⚙️ Technologies Used

- **PyTorch**
- **Hugging Face Transformers**
- **DistilBERT (pretrained model)**

---

## 🧠 Model Used

```text
distilbert-base-uncased-finetuned-sst-2-english
```

### Features

- Lightweight version of BERT
- Fine-tuned on SST-2 dataset
- Optimized for binary sentiment classification

---

## 📂 Project Structure

```text
pretrained-bert-model/
├── main.py
├── bert_model.py
└── README.md
```

---

## ▶️ How to Run

```bash
cd sentiment-analysis/pretrained-bert-model

python main.py
```

---

## 💡 How It Works

### Pipeline

```text
Input text
      │
      ▼
Tokenization (handled internally)
      │
      ▼
Pretrained BERT model
      │
      ▼
Prediction
```

### Model Outputs

- **Label** *(POSITIVE / NEGATIVE)*
- **Confidence score**

---

## 📊 Example

### Input

```text
I love this product
```

### Output

```text
Label : POSITIVE
Score : 0.999
```

---

## 🔍 Strengths

- ✔️ Understands context *(e.g. "not bad" → Positive)*
- ✔️ Works well on natural language
- ✔️ No training required

---

## ⚠️ Limitations

- Struggles with sarcasm
- Limited to training domain *(reviews-like text)*
- Binary classification only *(no neutral)*

---

## 📚 Key Learning Outcomes

- ✔️ Understand pretrained Transformer models
- ✔️ Learn inference pipeline with Hugging Face
- ✔️ Explore contextual NLP vs rule-based NLP

---

## 👨‍💻 Author

**Mohamed Amine REBAI**

*Software Engineering Student | AI Enthusiast*