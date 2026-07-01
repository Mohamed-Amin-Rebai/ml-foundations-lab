import joblib
import torch
import torch.nn as nn

from dataset_loader import text_to_indices


# -------------------------
# Model
# -------------------------
class SpamClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,64)
        self.fc1 = nn.Linear(64,32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32,2)

    def forward(self, x):
        x = self.embedding(x)

        x = x.mean(dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# -------------------------
# Load Model
# -------------------------
vocab = joblib.load(
    "model/vocab.pkl"
)

model = SpamClassifier(
    len(vocab)
)

model.load_state_dict(
    torch.load(
        "model/spam_model.pth",
        weights_only=True
    )
)

model.eval()


# -------------------------
# Prediction
# -------------------------
def predict_message(message):
    indices = text_to_indices(
        message,
        vocab
    )

    tensor = torch.tensor(
        [indices],
        dtype=torch.long
    )

    with torch.no_grad():
        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    label = (
        "🚨 SPAM"
        if prediction.item() == 1
        else "✅ HAM"
    )

    return (
        label,
        confidence.item() * 100
    )


# -------------------------
# CLI
# -------------------------
def main():
    print("📩 PyTorch Spam Detector")
    print("------------------------")

    message = input(
        "\nEnter a message:\n> "
    )

    label, confidence = predict_message(
        message
    )

    print("\nPrediction:")
    print(label)
    print(
        f"Confidence: {confidence:.2f}%"
    )


if __name__ == "__main__":
    main()