import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep only needed columns
df = df[["v1", "v2"]]

# Rename columns
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Features / Target
X = df["message"]
y = df["label"]

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Save
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\n✅ Model saved")