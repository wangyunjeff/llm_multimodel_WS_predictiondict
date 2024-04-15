import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class WindTurbineDataset(Dataset):
    def __init__(self, csv_file, seq_length, columns=None, split='train', train_split=0.8):
        """
        Args:
            csv_file (string): Path to the CSV file with data.
            seq_length (int): Number of time steps in each input sequence.
            columns (list of str, optional): List of column names to include in the dataset. If None, all columns are used.
            split (str): 'train' for training data, 'test' for testing data.
            train_split (float): Fraction of data to be used for training.
        """
        # Load and preprocess data
        data = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        # Select columns
        if columns is not None:
            data = data[columns]

        data = data.interpolate().values  # Interpolate and convert to NumPy array

        # Normalize data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = self.scaler.fit_transform(data)

        # Create sequences
        self.X, self.y = self.create_sequences(self.data, seq_length)

        # Determine split indices
        self.train_size = int(len(self.X) * train_split)
        self.split = split

    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length), :]
            y = data[i + seq_length, :]  # Supports multiple target variables
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def __len__(self):
        if self.split == 'train':
            return self.train_size
        else:
            return len(self.X) - self.train_size

    def __getitem__(self, idx):
        if self.split != 'train':
            idx += self.train_size
        sequence = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32).squeeze()  # Squeeze in case of single column
        return sequence, label


# Usage
csv_file = r'H:\Code_F\llm_multimodel_WS_predictiondict\data\la-haute-borne-data-2013-2016_new-3columns.csv'
seq_length = 20
columns = ['Ws_avg'] # P_avg, Ws_avg, Gost_avg

train_dataset = WindTurbineDataset(csv_file, seq_length, columns, 'train')
test_dataset = WindTurbineDataset(csv_file, seq_length, columns, 'test')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
