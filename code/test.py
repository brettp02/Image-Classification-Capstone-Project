import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
from train import ResNet
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    """
    Simple class to test accuracy of the RESNET50 model trained through transfer learning
    """
    # Check if CUDA is available and set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # normalization parameters
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # transformations for the test set (no data augmentations)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load test dataset
    test_data_dir = 'testdata'
    if not os.path.exists(test_data_dir):
        raise FileNotFoundError(f"Test data directory '{test_data_dir}' not found.")

    test_dataset = datasets.ImageFolder(test_data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize ResNet model
    model = ResNet()

    # Load the trained state dictionary
    model_path = 'model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # Lists to hold all true labels and predictions
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate accuracy
    correct = sum([1 for true, pred in zip(all_labels, all_predictions) if true == pred])
    total = len(all_labels)
    accuracy = 100 * correct / total
    print(f'\nResNet Accuracy: {accuracy:.2f}%\n')

    # Classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=test_dataset.classes))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=test_dataset.classes,
                yticklabels=test_dataset.classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == '__main__':
    main()
