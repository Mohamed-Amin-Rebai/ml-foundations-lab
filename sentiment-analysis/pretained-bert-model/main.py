from bert_model import analyze_bert


def main():
    print("🧠 BERT Sentiment Analysis")
    print("---------------------------")

    text = input("\nEnter your sentence:\n> ")

    result = analyze_bert(text)

    print("\n📊 Result")
    print("---------------------------")
    print(f"Label : {result['label']}")
    print(f"Score : {result['score']}")


if __name__ == "__main__":
    main()
