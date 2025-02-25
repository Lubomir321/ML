import time
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
from pprint import pprint


class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(0, 45)),
        transforms.RandomAutocontrast(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset_train = ImageFolder("clouds/clouds_train", transform=transform)
    dataset_test = ImageFolder("clouds/clouds_test", transform=transform)

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=16)
    dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=16)

    num_classes = len(dataset_train.classes)
    model = VGG16(num_classes=num_classes)
    optimizer = AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 2
    best_val_f1 = 0
    best_model_state = None

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        with tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", unit="batch") as tepoch:
            for images, labels in tepoch:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        train_loss /= len(dataloader_train)

        model.eval()
        val_labels = []
        val_preds = []
        val_loss = 0
        with tqdm(dataloader_test, desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]", unit="batch") as tepoch:
            for images, labels in tepoch:
                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_labels.extend(labels.numpy())
                    val_preds.extend(preds.numpy())
                tepoch.set_postfix(loss=loss.item())
        val_loss /= len(dataloader_test)
        val_f1 = f1_score(val_labels, val_preds, average='macro')

        print(f"Epoch {epoch + 1}/{num_epochs} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()

    end_time = time.time()
    print(f"Total training time: {end_time - start_time:.4f} seconds")

    model.load_state_dict(best_model_state)
    model.eval()
    test_labels = []
    test_preds = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader_test, desc="Testing", unit="batch"):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_labels.extend(labels.numpy())
            test_preds.extend(preds.numpy())

    precision = precision_score(test_labels, test_preds, average='macro')
    recall = recall_score(test_labels, test_preds, average='macro')
    f1 = f1_score(test_labels, test_preds, average='macro')
    f1_per_class = f1_score(test_labels, test_preds, average=None)

    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1: {f1:.4f}")
    print("Per-class F1 scores:")
    f1_dict = {dataset_test.classes[i]: round(score, 4) for i, score in enumerate(f1_per_class)}
    pprint(f1_dict)


if __name__ == "__main__":
    main()
