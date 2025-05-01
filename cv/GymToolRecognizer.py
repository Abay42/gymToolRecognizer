import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt
from tqdm import tqdm


NUM_CLASSES = 33
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 2e-5
LOWER_LR = 1e-5
FREEZE_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def create_balanced_sampler(dataset):
    labels = [label for _, label in dataset]
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


to_rgb = transforms.Lambda(lambda img: img.convert("RGB"))
base_transforms = [
    to_rgb,
    transforms.Resize((224, 224)),
]

train_transform = transforms.Compose(base_transforms + [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose(base_transforms + [
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_dir = 'dataset/images/train'
val_dir = 'dataset/images/val'
test_dir = 'dataset/images/test'

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)


sampler = create_balanced_sampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class GymToolRecognizer:
    def __init__(self, model_path="model.pth"):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.model.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
        self.model = self.model.to(DEVICE)

        log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        class_counts = np.bincount([label for _, label in train_dataset])
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
        class_weights = class_weights.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.1,
            patience=3,
            verbose=True
        )

        self.model_path = model_path
        self.train_accuracy = []
        self.val_accuracy = []
        self.test_accuracy = []
        self.test_loss = 0.0
        self.best_loss = float('inf')

        self._freeze_backbone()
        self.load_model()

    def _freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    def _unfreeze_last_layers(self):
        for name, param in self.model.named_parameters():
            if name.startswith("layer4.") or name.startswith("fc."):
                param.requires_grad = True
        trainable = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.Adam(trainable, lr=LOWER_LR)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            print(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            print(f"No saved model found at {self.model_path}, starting fresh.")

    def predict_image(self, image: Image.Image):
        self.model.eval()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted_class = torch.max(outputs, 1)

        return predicted_class.item()

    def train(self):
        print("Starting training...")
        prev_val_loss = None
        best_val_loss = float('inf')
        no_improve_count = 0
        patience = 3
        min_delta = 1e-3

        for epoch in range(1, EPOCHS + 1):
            if epoch == FREEZE_EPOCHS + 1:
                print("Unfreezing last layers and switching to lower LR")
                self._unfreeze_last_layers()
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    factor=0.1,
                    patience=3,
                    verbose=True
                )

            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)

            for images, labels in progress_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix(loss=loss.item())

            avg_train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            self.train_accuracy.append(train_acc)
            self.writer.add_scalar('Loss/train', avg_train_loss / len(train_loader), epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            print(f"Epoch {epoch}/{EPOCHS}, Accuracy: {train_acc * 100:.4f}%")

            # val
            self.model.eval()
            val_loss, val_preds, val_labels = 0.0, [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = self.model(images)
                    preds = outputs.argmax(dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = (np.array(val_preds) == np.array(val_labels)).mean()
            self.val_accuracy.append(val_acc)
            self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], epoch)
            print(f"Epoch {epoch}/{EPOCHS}, Val Acc: {val_acc * 100:.4f}%")

            if self.scheduler:
                self.scheduler.step(val_acc)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model()
                print("Checkpoint saved (best val loss).")

            if prev_val_loss is not None:
                if abs(prev_val_loss - avg_val_loss) < min_delta:
                    no_improve_count += 1
                    print(f"Val loss change < {min_delta}, no_improve_count = {no_improve_count}")
                    if no_improve_count >= patience:
                        print("Early stopping triggered.")
                        break
                else:
                    no_improve_count = 0
            prev_val_loss = avg_val_loss

            # test
            test_preds, test_labels = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = self.model(images)
                    preds = outputs.argmax(dim=1)
                    test_preds.extend(preds.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
            test_acc = (np.array(test_preds) == np.array(test_labels)).mean()
            self.test_accuracy.append(test_acc)
            print(f"Epoch {epoch}/{EPOCHS}, Test Acc: {test_acc * 100:.4f}%")

            if epoch % 10 == 0:
                self.save_model()
                print(f"Checkpoint saved at {self.model_path}")

        print(f"TensorBoard logs written to: {self.writer.log_dir}")
        self.writer.close()
        self.calculate_metrics()

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

    def calculate_metrics(self):
        self.model.eval()
        predictions, true_labels = [], []
        test_running_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        final_test_loss = test_running_loss / len(test_loader)
        final_test_accuracy = (np.array(predictions) == np.array(true_labels)).mean()

        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predictions)

        base_dir = "results"
        os.makedirs(base_dir, exist_ok=True)
        existing_folders = [int(name) for name in os.listdir(base_dir) if name.isdigit()]
        next_folder = str(max(existing_folders) + 1) if existing_folders else "1"
        save_dir = os.path.join(base_dir, next_folder)
        os.makedirs(save_dir, exist_ok=True)

        # Save metrics
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            f.write("\nFinal Test Metrics:\n")
            f.write(f"Test Loss: {final_test_loss:.6f}\n")
            f.write(f"Test Accuracy: {final_test_accuracy * 100:.4f}\n")
            f.write(f"Precision (Macro): {precision * 100:.4f}\n")
            f.write(f"Recall (Macro): {recall * 100:.4f}\n")
            f.write(f"F1-Score (Macro): {f1 * 100:.4f}\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n")

        print(f"Metrics saved to {save_dir}/metrics.txt")

        self.plot_accuracy(save_dir)

        print(f'Test Loss: {final_test_loss:.6f}')
        print(f'Test Accuracy: {final_test_accuracy * 100:.4f}')
        print(f'Precision (Macro): {precision * 100:.4f}')
        print(f'Recall (Macro): {recall * 100:.4f}')
        print(f'F1-Score (Macro): {f1 * 100:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')

    def plot_accuracy(self, save_dir):
        plt.figure(figsize=(10, 6))

        train_percentages = [a * 100 for a in self.train_accuracy]
        val_percentages = [a * 100 for a in self.val_accuracy]
        test_percentages = [a * 100 for a in self.test_accuracy]

        plt.plot(range(1, len(train_percentages) + 1), train_percentages, label='Training Accuracy', marker='o')
        plt.plot(range(1, len(val_percentages) + 1), val_percentages, label='Validation Accuracy', marker='s')
        plt.plot(range(1, len(test_percentages) + 1), test_percentages, label='Test Accuracy', marker='x')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.grid()

        plot_path = os.path.join(save_dir, 'accuracy_plot.png')
        plt.savefig(plot_path)
        print(f"Accuracy plot saved as '{plot_path}'")


if __name__ == "__main__":
    # print("\nDataset Statistics:")
    # print(f"Train samples: {len(train_dataset)}")
    # print(f"Val samples: {len(val_dataset)}")
    # print(f"Test samples: {len(test_dataset)}")
    #
    # # Class distribution analysis
    # train_counts = np.bincount([label for _, label in train_dataset])
    # print("\nTrain samples per class:")
    # print({train_dataset.classes[i]: count for i, count in enumerate(train_counts)})
    recognizer = GymToolRecognizer()
    recognizer.train()
