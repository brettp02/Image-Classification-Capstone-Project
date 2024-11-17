import torch
from torch import nn
import torchvision
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time
import os


class NeuralNet(nn.Module):
    """
    Custom CNN implementation based off of VGG architecture
    using 5 convlutions (3pix kernal size, with 1 pixel padding), with batch normalization, Max Pooling, and 3 fully connected layers going from 512 features to 3.
    Works with 224 x 224 size images, if changing image size adjust the (512 * x * x) values
    Uses Relu activation/non-linear function
    """
    def __init__(self):
        super(NeuralNet, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        # Block 1: Conv + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        # Block 2: Conv + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Block 3: Conv + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Block 4: Conv + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        # Block 5: Conv + BN + ReLU + Pool
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        # Flatten for fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ResNet(nn.Module):
    """
    Transfer Learning using the RESNET50 model pretrained on ImageNet data.
    Layers are frozen except the Fully connected layers for performance
    """
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet, self).__init__()
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=pretrained)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Freeze all layers except the final FC layer
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class EnsembleModel(nn.Module):
    """
    Stacks the custom CNN (VGG architecutre) with the RESNET50 model sums weighted outputs of both predictions
    """
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else torch.ones(len(models))
        self.weights = self.weights / self.weights.sum()  # Normalize weights

    def forward(self, x):
        outputs = []
        for model, weight in zip(self.models, self.weights):
            outputs.append(weight * model(x))
        return torch.stack(outputs).sum(0)


class ModelFactory:
    """
    Allows user to select which model to use for training
    model_type == 'neuralnet' || 'resnet' || 'ensemble' to select either of the above 3 NN's
    """
    @staticmethod
    def create_model(model_type, num_classes=3, **kwargs):
        if model_type.lower() == 'neuralnet':
            return NeuralNet()
        elif model_type.lower() == 'resnet':
            return ResNet(num_classes=num_classes)
        elif model_type.lower() == 'ensemble':
            models = [
                NeuralNet(),
                ResNet(num_classes=num_classes)
            ]
            weights = kwargs.get('weights', torch.ones(len(models)))
            return EnsembleModel(models, weights)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


class DataManager:
    """
    Class for dealing with loading data, splitting, preprocessing, and creating dataloaders
    batch_size = 32 was found to work best for performance and accuracy
    """
    def __init__(self, data_dir, batch_size=32, train_split=0.8):
        self.batch_size = batch_size
        self.train_split = train_split

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Train and test(validation) sets are augmented with random Horizontal flips to increase robustness of the model, this is not present in test.py and leads to greater accuracy on tests
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.dataset = datasets.ImageFolder(data_dir, transform=self.transform)
        self.class_names = self.dataset.classes
        self._create_data_loaders()

    def _create_data_loaders(self):
        """
        Creates dataloaders for train and test sets with an 80/20 split
        uses num_workers 4 for each to utilize parallisation
        """
        total_size = len(self.dataset)
        train_size = int(self.train_split * total_size)
        test_size = total_size - train_size

        train_data, test_data = random_split(self.dataset, [train_size, test_size])

        # Create data loaders
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

        self.test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )


class Trainer:
    """
    Class to deal with training loop, loss function, optimizer, and accuracy reporting/visualisation
    """
    def __init__(self, model, data_manager, device, optimizer_config=None):
        self.model = model.to(device)
        self.data_manager = data_manager
        self.device = device

        # Best performing optimzers for each of the models based on experimentation (no CV)
        default_config = {
            'neuralnet': {
                'type': 'adam',
                'lr': 0.001,
                'weight_decay': 1e-5
            },
            'resnet': {
                'type': 'sgd',
                'lr': 0.01,
                'momentum': 0.9
            },
            'ensemble': {
                'type': 'adam',
                'lr': 0.001,
                'weight_decay': 1e-5
            }
        }

        # Optionally manually confugure optimzer in main()
        optimizer_config = optimizer_config or default_config.get(
            type(model).__name__.lower(),
            default_config['neuralnet']
        )

        # CrossEntropyLoss function selected (works best for each model)
        self.loss_function = nn.CrossEntropyLoss()

        # Loads the selected optimizer based on model selected or manual input
        self.optimizer = self._create_optimizer(optimizer_config)

        # Scheduler to lower the learning rate by a factor of 0.1 each 7 steps
        self.scheduler = StepLR(self.optimizer, step_size=7, gamma=0.1)

        print(f"\nOptimizer Type: {type(self.optimizer).__name__}")

        # Init timers
        self.start_time = None
        self.epoch_start_time = None

    def _create_optimizer(self, config):
        """
        Creates optimizers (SGD and Adam variants)

        :param config: configuration
        :return: optimizer
        """
        if config['type'].lower() == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=config['lr'],
                weight_decay=config.get('weight_decay', 0)
            )
        elif config['type'].lower() == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=config['lr'],
                momentum=config.get('momentum', 0)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {config['type']}")

    def train(self, num_epochs=30):
        """
        Main train loop

        :param num_epochs: how many iterations to train the model
        """
        self.start_time = time.time()
        best_val_accuracy = 0.0
        metrics = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'epoch_times': []
        }

        for epoch in range(num_epochs):
            self.epoch_start_time = time.time()

            # Training phase (gets loss and accuracy values, and appends to related metrics)
            train_loss, train_acc = self._train_epoch()
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)

            # Validation phase
            val_loss, val_acc, labels, preds = self._validate()
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)

            # Calculate epoch time
            epoch_time = time.time() - self.epoch_start_time
            metrics['epoch_times'].append(epoch_time)

            # Step learning rate scheduler
            self.scheduler.step()

            # Print epoch results
            self._print_epoch_results(epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, epoch_time)

            # Save best model (use best_model as name to avoid overwriting current best .pth file)
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f"Saved best model with accuracy: {best_val_accuracy:.2f}%")

        # Calculate and print total training time
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}")

        print(f'Best val accuracy: {best_val_accuracy}')
        # Print final metrics
        self._print_final_metrics(labels, preds)

        # Plot training curves
        self._plot_training_curves(metrics, num_epochs)

    def _train_epoch(self):
        """
        Calculate loss function and accuracy for each epoch
        each epoch resets gradients, calculates loss functions, back propagates and uses optimizer
        """

        # Set model to training mode and instantiate fields
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training Loop
        for inputs, labels in self.data_manager.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward Pass
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)

            # Backprop and optimization
            loss.backward()
            self.optimizer.step()

            # Track accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.data_manager.train_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def _validate(self):
        """
        Sets model to evaluation mode for validation set
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in self.data_manager.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_function(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(self.data_manager.test_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc, all_labels, all_preds

    def _calculate_per_class_accuracy(self, labels, preds):
        """
        Helper function to see the accuracy of each class
        :param labels: class name
        :param preds: predicted values
        :return: accuracy per class
        """
        per_class_acc = {}
        for class_idx, class_name in enumerate(self.data_manager.class_names):
            mask = (np.array(labels) == class_idx)
            if mask.sum() > 0:
                class_acc = accuracy_score(
                    np.array(labels)[mask],
                    np.array(preds)[mask]
                )
                per_class_acc[class_name] = class_acc * 100
        return per_class_acc

    def _print_epoch_results(self, epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, epoch_time):
        """
        Print results for each epoch (train acc/loss, val acc/loss), best model is selected off of best val acc
        """
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_time:.2f}s"
        )

    def _print_final_metrics(self, labels, preds):
        """
        Print final metrics and display report with performance metrics
        """
        print("\n" + "=" * 50)
        print("FINAL TRAINING METRICS")
        print("=" * 50)

        # Print general classification report
        print("\nDetailed Classification Report:")
        print(classification_report(labels, preds, target_names=self.data_manager.class_names))

        # Print per-class accuracies
        print("\nPer-Class Accuracies:")
        per_class_acc = self._calculate_per_class_accuracy(labels, preds)
        for class_name, accuracy in per_class_acc.items():
            print(f"{class_name}: {accuracy:.2f}%")

    def _plot_training_curves(self, metrics, num_epochs):
        """
        Visualtion
        """

        # Plot Loss
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(range(1, num_epochs + 1), metrics['train_loss'], label='Training Loss')
        plt.plot(range(1, num_epochs + 1), metrics['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(range(1, num_epochs + 1), metrics['train_acc'], label='Training Accuracy')
        plt.plot(range(1, num_epochs + 1), metrics['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        # Plot Epoch Times
        plt.subplot(1, 3, 3)
        plt.plot(range(1, num_epochs + 1), metrics['epoch_times'])
        plt.title('Epoch Training Times')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')

        plt.tight_layout()
        plt.show()


def main():
    """
    Main Method:
    - Dynamically sets device ('cuda', 'cpu')
    - Loads and processes data
    - Selects and creates model
    - Trains model for selected num_epochs
    """
    print("Starting training process...")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # Initialize data manager
    print("Initializing data manager...")
    data_manager = DataManager('../datasets/train_data_clean_enriched')
    print(f"Found {len(data_manager.class_names)} classes: {data_manager.class_names}")

    # Choose model type ('neuralnet', 'resnet', or 'ensemble')
    model_type = 'resnet'

    # Initialize model
    print(f"Initializing {model_type} model...")
    model = ModelFactory.create_model(
        model_type,
        num_classes=len(data_manager.class_names),
    )

    trainer = Trainer(model, data_manager, device)

    # Start training
    print("Starting training...")
    trainer.train(num_epochs=30)


if __name__ == '__main__':
    main()