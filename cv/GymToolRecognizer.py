import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_CLASSES = 33
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_dir = 'dataset/images/train'
val_dir = 'dataset/images/val'
test_dir = 'dataset/images/test'

train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
test_dataset = datasets.ImageFolder(root=test_dir)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class GymToolRecognizer:
    def __init__(self, model_path="model.pth"):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)
        self.model = self.model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)
        self.model_path = model_path
        self.train_accuracy = []
        self.val_accuracy = []
        self.test_accuracy = 0.0
        self.test_loss = 0.0

        self.load_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            self.model.eval()
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
        self.model.train()

        for epoch in range(EPOCHS):
            running_loss = 0.0
            correct = 0
            total = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
            for images, labels in progress_bar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix(loss=loss.item())

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            train_acc = correct / total
            self.train_accuracy.append(train_acc)
            print(f"Epoch {epoch + 1}/{EPOCHS}, Accuracy: {train_acc * 100:.4f}")

            val_acc = self.evaluate(val_loader)
            self.val_accuracy.append(val_acc)
            print(f"Epoch {epoch + 1}/{EPOCHS}, Validation Accuracy: {val_acc * 100:.4f}")

        self.save_model()
        self.calculate_metrics()
        self.plot_accuracy()

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
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_running_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        self.test_loss = test_running_loss / len(test_loader)
        self.test_accuracy = self.evaluate(test_loader)

        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(true_labels, predictions)

        with open("results/metrics.txt", "w") as f:
            f.write("\nFinal Test Metrics:\n")
            f.write(f"Test Loss: {self.test_loss:.6f}\n")
            f.write(f"Test Accuracy: {self.test_accuracy*100:.4f}\n")
            f.write(f"Precision (Macro): {precision*100:.4f}\n")
            f.write(f"Recall (Macro): {recall*100:.4f}\n")
            f.write(f"F1-Score (Macro): {f1*100:.4f}\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n")

        print(f"Metrics saved to metrics.txt")
        print(f'Test Loss: {self.test_loss:.6f}')
        print(f'Test Accuracy: {self.test_accuracy*100:.4f}')
        print(f'Precision (Macro): {precision*100:.4f}')
        print(f'Recall (Macro): {recall*100:.4f}')
        print(f'F1-Score (Macro): {f1*100:.4f}')
        print(f'Confusion Matrix:\n{conf_matrix}')

    def plot_accuracy(self):
        plt.figure(figsize=(10, 6))

        train_percentages = [a * 100 for a in self.train_accuracy]
        val_percentages = [a * 100 for a in self.val_accuracy]
        test_percentage = self.test_accuracy * 100

        plt.plot(list(range(1, len(self.train_accuracy) + 1)), train_percentages, label='Training Accuracy', marker='o')
        plt.plot(list(range(1, len(self.val_accuracy) + 1)), val_percentages, label='Validation Accuracy', marker='s')
        plt.axhline(y=test_percentage, color='r', linestyle='--', label='Test Accuracy')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Over Epochs')
        plt.legend()
        plt.grid()
        plt.savefig('results/accuracy_plot.png')
        print("Accuracy plot saved as 'training_accuracy_plot.png'")


if __name__ == "__main__":
    recognizer = GymToolRecognizer()
    recognizer.train()
