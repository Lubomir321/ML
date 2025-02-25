import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from task1 import WaterDataset
from tqdm import tqdm

torch.manual_seed(42)


class ImprovedNetV2(nn.Module):
    def __init__(self):
        super(ImprovedNetV2, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)

        self.fc4 = nn.Linear(16, 1)

        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = nn.functional.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = nn.functional.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = nn.functional.leaky_relu(self.bn3(self.fc3(x)))
        x = torch.sigmoid(self.fc4(x))
        return x



def train_model_v2(dataloader_train, dataloader_val, optimizer, net, num_epochs=100):
    criterion = nn.BCELoss()
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        net.train()
        total_train_loss = 0
        with tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as train_bar:
            for features, labels in train_bar:
                features = features.float()
                labels = labels.float().view(-1, 1)

                optimizer.zero_grad()
                outputs = net(features)
                loss = criterion(outputs, labels)
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()

                train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)

        #validation
        net.eval()
        total_val_loss = 0
        with tqdm(dataloader_val, desc=f"Validation {epoch+1}/{num_epochs}", unit="batch") as val_bar:
            with torch.no_grad():
                for features, labels in val_bar:
                    features = features.float()
                    labels = labels.float().view(-1, 1)
                    outputs = net(features)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

                    val_bar.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(dataloader_val)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), "best_model.pth")

    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()


def evaluate_model(dataloader, net):
    y_true = []
    y_pred = []
    net.eval()
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.float()
            labels = labels.float()
            outputs = net(features)
            predictions = (outputs > 0.5).float()
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.numpy())

    f1 = f1_score(y_true, y_pred)
    return f1

def main():
    full_dataset = WaterDataset("water_train.csv")
    test_dataset = WaterDataset("water_test.csv")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    net = ImprovedNetV2()
    optimizer = optim.AdamW(net.parameters(), lr=0.001)
    train_model_v2(train_loader, val_loader, optimizer, net, num_epochs=1500) #tried with more but the loss was lingering btween 0.7-0.73

    net.load_state_dict(torch.load("best_model.pth"))

    f1_test = evaluate_model(test_loader, net)
    print(f"F1 score on test set: {f1_test:.6f}")

if __name__ == "__main__":
    main()