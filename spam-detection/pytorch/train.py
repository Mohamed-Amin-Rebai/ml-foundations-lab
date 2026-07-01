import pandas as pd
import joblib
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset_loader import SpamDataset, build_vocab


# -------------------------
# Load Data
# -------------------------
df = pd.read_csv(
    "dataset/spam.csv",
    encoding="latin-1"
)

df = df[["v1", "v2"]]
df.columns = ["label", "message"]

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# -------------------------
# Vocabulary
# -------------------------
vocab = build_vocab(df["message"])


# -------------------------
# Split
# -------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)


# -------------------------
# Datasets
# -------------------------
train_dataset = SpamDataset(
    train_df,
    vocab
)

test_dataset = SpamDataset(
    test_df,
    vocab
)


# -------------------------
# DataLoaders
# -------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32
)


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


model = SpamClassifier(len(vocab))


# -------------------------
# Training Setup
# -------------------------
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------
# Training Loop
# -------------------------
epochs = 5

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for messages, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(messages)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss:.4f}"
    )


# -------------------------
# Evaluation
# -------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for messages, labels in test_loader:

        outputs = model(messages)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)


accuracy = correct / total
print(f"\nAccuracy: {accuracy:.4f}")


# -------------------------
# Save
# -------------------------
torch.save(
    model.state_dict(),
    "model/spam_model.pth"
)

joblib.dump(
    vocab,
    "model/vocab.pkl"
)

print("\n✅ Model saved")