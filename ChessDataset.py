import pandas as pd
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, df):
        super().__init__()
        # Convert data to a NumPy array and assign to self.data
        self.data = df.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        # Assign last data column to label
        label = self.data[idx, -1]
        return features, label