# 🧠 Sentiment Analysis

> This module implements **rule-based sentiment analysis** using:
>
> - ✅ TextBlob
> - ✅ VADER
>
> It compares both approaches on the same input to highlight differences in sentiment detection.

---

## 🎯 Goal

Classify a given sentence into:

- ✅ Positive
- ❌ Negative
- ➖ Neutral

And compare results between two methods.

---

## ⚙️ Technologies Used

- **TextBlob** → General-purpose sentiment analysis
- **VADER** → Optimized for social media / informal text

---

## 📂 Project Structure

```text
simple/
├── main.py
├── textblob_model.py
├── vader_model.py
└── README.md
```

---

## ▶️ How to Run

```bash
cd sentiment-analysis/rule-based-nlp

python main.py
```

---

## 💡 How It Works

### 🔹 TextBlob

Returns a **polarity score** between **-1** and **1**.

#### Interpretation

```text
> 0  → Positive
< 0  → Negative
= 0  → Neutral
```

---

### 🔹 VADER

Returns:

```python
{
    "neg": ...,
    "neu": ...,
    "pos": ...,
    "compound": ...
}
```

Uses the **compound score** for final sentiment:

```text
≥ 0.05   → Positive
≤ -0.05  → Negative
otherwise → Neutral
```

---

## 📊 Example

### Input

```text
I love this product!!!
```

### Output

```text
TextBlob

Label    : Positive
Polarity : 0.8


VADER

Label     : Positive
Compound  : 0.85
Details   : { ... }


Comparison

✅ Models agree
```

---

## 📚 Key Learning Outcomes

- ✔️ Understand rule-based sentiment analysis
- ✔️ Learn polarity vs compound scores
- ✔️ Compare two NLP approaches
- ✔️ Identify limitations of simple models

---

## 🔍 Observations

### VADER performs better on:

- 😊 Emojis
- Slang ("soooo good!!!")
- Capitalization

### TextBlob is:

- Simpler
- More general-purpose

---

## 👨‍💻 Author

**Mohamed Amine REBAI**

*Software Engineering Student | AI Enthusiast*

<!-- 
test :
I love this product
This is amazing
I hate this
This is terrible
This is okay
This is not bad
I expected better
It's fine I guess
That was kinda disappointing
I LOVE THIS!!! 😍🔥
this is sooo good!!!
meh...
wow... just wow 😒 
Yeah great... just what I needed
Oh nice, another bug...
Love when things break 😊
-->