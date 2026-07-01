from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize once (important)
analyzer = SentimentIntensityAnalyzer()


def analyze_vader(text):
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "compound": round(compound, 3),
        "details": scores
    }