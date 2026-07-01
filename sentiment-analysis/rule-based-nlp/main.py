from textblob_model import analyze_textblob
from vader_model import analyze_vader


# ---------------------------
# Compare Results
# ---------------------------
def compare_results(tb_result, vd_result):
    if tb_result["label"] == vd_result["label"]:
        return "✅ Models agree"
    else:
        return "⚠️ Models DISAGREE"


# ---------------------------
# Main CLI App
# ---------------------------
def main():
    print("🧠 Sentiment Analysis (TextBlob vs VADER)")
    print("----------------------------------------")

    text = input("\nEnter your sentence:\n> ")

    tb_result = analyze_textblob(text)
    vd_result = analyze_vader(text)

    print("\n📊 Results")
    print("----------------------------------------")

    print("\n🔹 TextBlob")
    print(f"Label    : {tb_result['label']}")
    print(f"Polarity : {tb_result['polarity']}")

    print("\n🔹 VADER")
    print(f"Label     : {vd_result['label']}")
    print(f"Compound  : {vd_result['compound']}")
    print(f"Details   : {vd_result['details']}")

    print("\n🔍 Comparison")
    print(compare_results(tb_result, vd_result))


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    main()