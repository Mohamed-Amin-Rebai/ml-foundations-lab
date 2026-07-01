import joblib


# Load model and vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def predict_message(message):
    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)[0]

    if prediction == 1:
        return "🚨 SPAM"
    else:
        return "✅ HAM"


def main():
    print("📩 Spam Detection System")
    print("------------------------")

    message = input("\nEnter a message:\n> ")

    result = predict_message(message)

    print("\nPrediction:")
    print(result)


if __name__ == "__main__":
    main()