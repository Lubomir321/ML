import time
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomApply, RandomAutocontrast, Resize, ToTensor
from torch import nn
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
from pprint import pprint
from tqdm import tqdm 

class CloudCNN(nn.Module):
    def __init__(self, num_classes):
        super(CloudCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
                
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2),
        )
            
        self.classifier = nn.Linear(64 * 16 * 16, num_classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    transform = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(degrees=(0, 45)),
        RandomAutocontrast(),
        ToTensor(),
        Resize((64, 64))
    ])

    dataset_train = ImageFolder("clouds/clouds_train", transform=transform)
    dataset_test = ImageFolder("clouds/clouds_test", transform=transform)

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=16)
    dataloader_test = DataLoader(dataset_test, shuffle=False, batch_size=16)


    model = CloudCNN(len(dataset_train.classes))
    optimizer = AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        with tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as tepoch:
            for images, labels in tepoch:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
        print(f"Average training loss per epoch: {epoch_loss / len(dataloader_train):.4f}")

    end_time = time.time()
    print(f"Total time taken to train the model in seconds: {end_time - start_time:.4f}")

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

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print("Per-class F1 scores:")
    f1_dict = {dataset_test.classes[i]: round(score, 4) for i, score in enumerate(f1_per_class)}
    pprint(f1_dict)


    #The training process took 47.5433 seconds, which is acceptable given the dataset size and model complexity.
    #The global F1 score 0.5054 which is not great. Some classes, such as cumulonimbus clouds have a significantly lower F1 score.
    #The clear sky class has the highest F1 score, indicating the model performs well for this class.
    #A recall of 0.51 means that mode is missing true positives.
    #A precision of 0.61 indicates that the model makes moderately accurate predictions.

if __name__ == "__main__":
    main()