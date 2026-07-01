from textblob import TextBlob


def analyze_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        label = "Positive"
    elif polarity < 0:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "polarity": round(polarity, 3)
    }