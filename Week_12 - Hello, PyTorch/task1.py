import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class WaterDataset(Dataset):
    def __init__(self, csv_file):
        super().__init__()
        data = pd.read_csv(csv_file).to_numpy()
        self.features = data[:, :-1].astype(np.float64)
        self.labels = data[:, -1].astype(np.float64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

def main():
    dataset = WaterDataset("water_train.csv")

    print(dataset.__len__())

    print(dataset.__getitem__(4))

    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    first_batch = next(iter(train_dataloader))
    features, labels = first_batch
    print(features, labels)

if __name__ == "__main__":
    main()