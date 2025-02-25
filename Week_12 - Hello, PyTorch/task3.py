import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from task1 import WaterDataset

torch.manual_seed(42)


class ImprovedNet(nn.Module):
    def __init__(self):
        super(ImprovedNet, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)
        
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = nn.functional.elu(self.bn1(self.fc1(x)))
        x = nn.functional.elu(self.bn2(self.fc2(x)))
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
    test_dataset = WaterDataset("water_test.csv")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    optimizers = {
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
    }

    for opt_name, opt_class in optimizers.items():
        print(f"Using the {opt_name} optimizer:")
        net = ImprovedNet()
        optimizer = opt_class(net.parameters(), lr=0.001)
        losses = train_model(train_loader, optimizer, net, num_epochs=10)
        print(f"Average loss: {losses:.6f}")

    print("\nTraining for 1000 epochs with AdamW...")
    net = ImprovedNet()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    losses = train_model(train_loader, optimizer, net, num_epochs=1000, create_plot=True)

    y_true = []
    y_pred = []

    net.eval()
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.float()
            labels = labels.float()
            outputs = net(features)
            predictions = (outputs > 0.5).float()
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

    f1 = f1_score(y_true, y_pred)
    print(f"F1 score on test set: {f1:.6f}")

    #C. Batch normalization effectively learns the optimal input distribution for each layer it precedes.
if __name__ == "__main__":
    main()