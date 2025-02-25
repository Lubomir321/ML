import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils import data
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn


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
    train_transforms = transforms.Compose([  
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Resize((64,64))
    ])

    dataset_train = ImageFolder('clouds/clouds_train', transform=train_transforms)

    image, label = dataset_train[0]
    plt.imshow(image.permute(1, 2, 0))
    plt.title(f"Class: {dataset_train.classes[label]}")
    plt.axis('off')
    plt.show()

    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)

    num_classes = len(dataset_train.classes)
    model = CloudCNN(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.001)

    epochs = 20

    losses_per_epoch = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for images, labels in loop:

            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(train_loader)
        losses_per_epoch.append(avg_loss)
        print(f"Average training loss per epoch: {avg_loss:.4f}")


    plt.figure(figsize=(8, 6))
    plt.plot(losses_per_epoch, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
