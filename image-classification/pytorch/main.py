import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


# -------------------------
# Classes
# -------------------------
classes = ["Cat", "Dog"]


# -------------------------
# Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# -------------------------
# Model
# -------------------------
model = models.resnet18()

model.fc = nn.Linear(
    model.fc.in_features,
    2
)

model.load_state_dict(
    torch.load(
        "model/cats_dogs_model.pth",
        weights_only=True
    )
)

model.eval()


# -------------------------
# Prediction
# -------------------------
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    label = classes[
        prediction.item()
    ]

    return (
        label,
        confidence.item() * 100
    )


# -------------------------
# CLI
# -------------------------
def main():
    print("🐶🐱 Cats vs Dogs Classifier")
    print("---------------------------")

    image_path = input(
        "\nEnter image path:\n> "
    )

    label, confidence = predict_image(
        image_path
    )

    emoji = "🐱" if label == "Cat" else "🐶"

    print("\nPrediction:")
    print(f"{emoji} {label}")
    print(
        f"Confidence: {confidence:.2f}%"
    )


if __name__ == "__main__":
    main()