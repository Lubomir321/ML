import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from task1 import WaterDataset

torch.manual_seed(42)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train_model(dataloader_train: DataLoader, optimizer, net, num_epochs: int, create_plot: bool = False):
    criterion = nn.BCELoss()
    epoch_losses = []

    with tqdm(total=num_epochs * len(dataloader_train), desc="Training Progress", unit="batch") as pbar:
        for epoch in range(num_epochs):
            total_loss = 0

            for features, labels in dataloader_train:
                features = features.float()
                labels = labels.float().view(-1, 1)

                optimizer.zero_grad()
                outputs = net(features)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(epoch=epoch + 1, loss=loss.item())

            avg_loss = total_loss / len(dataloader_train)
            epoch_losses.append(avg_loss)

    if create_plot:
        plt.plot(range(1, num_epochs + 1), epoch_losses, label="Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.show()

    return sum(epoch_losses) / len(epoch_losses)

def main():
    train_dataset = WaterDataset("water_train.csv")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    net = Net()

    optimizers = {
        "SGD": optim.SGD(net.parameters(), lr=0.001),
        "RMSprop": optim.RMSprop(net.parameters(), lr=0.001),
        "Adam": optim.Adam(net.parameters(), lr=0.001),
        "AdamW": optim.AdamW(net.parameters(), lr=0.001),
    }

    for opt_name, optimizer in optimizers.items():
        print(f"Using the {opt_name} optimizer:")
        net = Net()
        losses = train_model(train_loader, optimizer, net, num_epochs=10)
        print(f"Average loss: {np.mean(losses)}")

    print("\nTraining for 1000 epochs with AdamW...")
    net = Net()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    losses = train_model(train_loader, optimizer, net, num_epochs=1000, create_plot=True)

    test_dataset = WaterDataset("water_test.csv")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    y_true = []
    y_pred = []

    net.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.float()
            labels = labels.float()
            #one was double
            #RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
            outputs = net(features)
            predictions = (outputs > 0.5).float()
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

    f1 = f1_score(y_true, y_pred)
    print(f"F1 score on test set: {f1}")

    #The F1 score is not very high, indicating that the model struggles to make accurate predictions.

if __name__ == "__main__":
    main()