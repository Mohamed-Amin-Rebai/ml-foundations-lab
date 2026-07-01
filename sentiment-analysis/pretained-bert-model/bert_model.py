from transformers import pipeline

# Load pretrained sentiment model
classifier = pipeline("sentiment-analysis")


def analyze_bert(text):
    result = classifier(text)[0]

    return {
        "label": result["label"],
        "score": round(result["score"], 3)
    }