import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from PIL import Image
import os


# -------------------------
# Transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# Remove Bad Files
# -------------------------
bad_files = []

for folder in [
    "dataset/PetImages/Cat",
    "dataset/PetImages/Dog"
]:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        try:
            img = Image.open(path)
            img.verify()
        except:
            bad_files.append(path)

print("Bad files:", len(bad_files))

for file in bad_files:
    os.remove(file)


# -------------------------
# Dataset
# -------------------------
dataset = datasets.ImageFolder(
    "dataset/PetImages",
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# -------------------------
# Split
# -------------------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size]
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
model = models.resnet18(weights="DEFAULT")

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# -------------------------
# Training Setup
# -------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# -------------------------
# Train
# -------------------------
epochs = 3

for epoch in range(epochs):
    model.train()

    running_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f}"
    )


# -------------------------
# Test
# -------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

accuracy = correct / total

print(f"\nAccuracy: {accuracy:.4f}")


# -------------------------
# Save
# -------------------------
torch.save(
    model.state_dict(),
    "model/cats_dogs_model.pth"
)

print("\n✅ Model saved")