import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

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
        # self.scaler = MinMaxScaler(feature_range=(0, 1))
        # self.data = self.scaler.fit_transform(data)

        self.data = data

        # Create sequences
        self.X, self.y = self.create_sequences(self.data, seq_length)

        # Determine split index
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
        if self.split == 'test':
            idx += self.train_size
        sequence = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32).squeeze()  # Squeeze in case of single column
        return sequence, label


if __name__ == '__main__':

    # Example usage
    csv_file = 'sorted_resample_la-haute-borne-data-2013-2016_1column.csv'
    seq_length = 24  # Example sequence length

    train_dataset = WindTurbineDataset(csv_file, seq_length, split='train')
    test_dataset = WindTurbineDataset(csv_file, seq_length, split='test')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    for sequence, label in train_dataloader:
        print(sequence.shape, label.shape)
        break

    for sequence, label in test_dataloader:
        print(sequence.shape, label.shape)
        break


# Usage
# csv_file = r'H:\Code_F\llm_multimodel_WS_predictiondict\data\la-haute-borne-data-2013-2016_new-3columns.csv'
# seq_length = 20
# columns = ['Ws_avg'] # P_avg, Ws_avg, Gost_avg
#
# train_dataset = WindTurbineDataset(csv_file, seq_length, columns, 'train')
# test_dataset = WindTurbineDataset(csv_file, seq_length, columns, 'test')
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import pandas as pd
#
#
# class WindSpeedDataset(Dataset):
#     def __init__(self, surface_file, upper_file, wind_speed_file, seq_len=6):
#         # Load the data
#         self.surface_data = np.load(surface_file)  # Shape: (4, 721, 1440)
#         self.upper_data = np.load(upper_file)  # Shape: (5, 13, 721, 1440)
#         self.wind_speed_data = np.load(wind_speed_file)  # Shape: (number_of_samples, 1)
#
#         # Ensure the data length matches for synchronization
#         self.seq_len = seq_len
#         self.num_hours = self.surface_data.shape[1]  # Assuming hourly data
#         self.num_ten_min_intervals = self.wind_speed_data.shape[0]
#
#         # Verify data consistency
#         assert self.num_hours * 6 == self.num_ten_min_intervals, "Data length mismatch"
#
#     def __len__(self):
#         return self.num_hours - self.seq_len + 1
#
#     def __getitem__(self, idx):
#         # Surface and upper-air data for the sequence
#         surface_seq = self.surface_data[:, idx:idx + self.seq_len, :]
#         upper_seq = self.upper_data[:, :, idx:idx + self.seq_len, :]
#
#         # Repeat the hourly data to match the 10-minute interval data
#         surface_seq_repeated = np.repeat(surface_seq, 6, axis=1)
#         upper_seq_repeated = np.repeat(upper_seq, 6, axis=2)
#
#         # Corresponding wind speed data for the sequence
#         wind_speed_seq = self.wind_speed_data[idx * 6:(idx + self.seq_len) * 6, :]
#
#         return (
#             torch.tensor(surface_seq_repeated, dtype=torch.float32),
#             torch.tensor(upper_seq_repeated, dtype=torch.float32),
#             torch.tensor(wind_speed_seq, dtype=torch.float32)
#         )
#
#
# class WindTurbineDataset(Dataset):
#     def __init__(self, csv_file, seq_length, columns=None, split='train', train_split=0.8):
#         """
#         Args:
#             csv_file (string): Path to the CSV file with data.
#             seq_length (int): Number of time steps in each input sequence.
#             columns (list of str, optional): List of column names to include in the dataset. If None, all columns are used.
#             split (str): 'train' for training data, 'test' for testing data.
#             train_split (float): Fraction of data to be used for training.
#         """
#         # Load and preprocess data
#         data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
#
#         # Select columns
#         if columns is not None:
#             data = data[columns]
#
#         data = data.interpolate().values  # Interpolate and convert to NumPy array
#
#         # Normalize data (if necessary)
#         # self.scaler = MinMaxScaler(feature_range=(0, 1))
#         # self.data = self.scaler.fit_transform(data)
#
#         self.data = data
#
#         # Create sequences
#         self.X, self.y = self.create_sequences(self.data, seq_length)
#
#         # Determine split index
#         self.train_size = int(len(self.X) * train_split)
#         self.split = split
#
#     def create_sequences(self, data, seq_length):
#         xs, ys = [], []
#         for i in range(len(data) - seq_length):
#             x = data[i:(i + seq_length), :]
#             y = data[i + seq_length, :]  # Supports multiple target variables
#             xs.append(x)
#             ys.append(y)
#         return np.array(xs), np.array(ys)
#
#     def __len__(self):
#         if self.split == 'train':
#             return self.train_size
#         else:
#             return len(self.X) - self.train_size
#
#     def __getitem__(self, idx):
#         if self.split == 'test':
#             idx += self.train_size
#         sequence = torch.tensor(self.X[idx], dtype=torch.float32)
#         label = torch.tensor(self.y[idx], dtype=torch.float32).squeeze()  # Squeeze in case of single column
#         return sequence, label
#
#
# class CombinedDataset(Dataset):
#     def __init__(self, surface_file, upper_file, wind_speed_file, csv_file, seq_len=6, seq_length_turbine=6,
#                  columns=None, split='train', train_split=0.8):
#         self.wind_speed_dataset = WindSpeedDataset(surface_file, upper_file, wind_speed_file, seq_len)
#         self.wind_turbine_dataset = WindTurbineDataset(csv_file, seq_length_turbine, columns, split, train_split)
#         self.split = split
#
#         if self.split == 'train':
#             self.len = min(len(self.wind_speed_dataset), len(self.wind_turbine_dataset))
#         else:
#             self.len = min(len(self.wind_speed_dataset), len(self.wind_turbine_dataset))
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         wind_speed_data = self.wind_speed_dataset[idx]
#         wind_turbine_data = self.wind_turbine_dataset[idx]
#
#         return wind_speed_data, wind_turbine_data
#
#
# # Example usage
# surface_file = 'input_surface.npy'
# upper_file = 'input_upper.npy'
# wind_speed_file = 'wind_speed.npy'
# csv_file = 'path_to_your_csv_file.csv'
# seq_len = 6
# seq_length_turbine = 6
#
# combined_dataset = CombinedDataset(surface_file, upper_file, wind_speed_file, csv_file, seq_len, seq_length_turbine)
# combined_dataloader = DataLoader(combined_dataset, batch_size=8, shuffle=True)
#
# for (surface, upper, wind_speed), (turbine_seq, turbine_label) in combined_dataloader:
#     print(surface.shape, upper.shape, wind_speed.shape)
#     print(turbine_seq.shape, turbine_label.shape)
#     break
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class WindTurbineDataset(Dataset):
    def __init__(self, csv_file, input_surface_file, input_upper_file, seq_length, columns=None, split='train',
                 train_split=0.8):
        """
        Args:
            csv_file (string): Path to the CSV file with wind speed time series data.
            input_surface_file (string): Path to the numpy file with surface variables data.
            input_upper_file (string): Path to the numpy file with upper-air variables data.
            seq_length (int): Number of time steps in each input sequence.
            columns (list of str, optional): List of column names to include in the dataset. If None, all columns are used.
            split (str): 'train' for training data, 'test' for testing data.
            train_split (float): Fraction of data to be used for training.
        """
        # Load and preprocess wind speed time series data
        self.wind_speed_data = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        if columns is not None:
            self.wind_speed_data = self.wind_speed_data[columns]

        self.wind_speed_data = self.wind_speed_data.interpolate().values  # Interpolate and convert to NumPy array

        # Load surface and upper-air variables data
        self.surface_data = np.load(input_surface_file)
        self.upper_data = np.load(input_upper_file)

        # Create sequences
        self.seq_length = seq_length
        self.X_surface, self.X_upper, self.X_wind, self.y = self.create_sequences(self.surface_data, self.upper_data,
                                                                                  self.wind_speed_data, seq_length)

        # Determine split index
        self.train_size = int(len(self.X_wind) * train_split)
        self.split = split

    def create_sequences(self, surface_data, upper_data, wind_data, seq_length):
        xs_surface, xs_upper, xs_wind, ys = [], [], [], []
        for i in range(len(wind_data) - seq_length):
            x_surface = surface_data[:, i:(i + seq_length)]
            x_upper = upper_data[:, :, i:(i + seq_length)]
            x_wind = wind_data[i:(i + seq_length), :]
            y = wind_data[i + seq_length, :]
            xs_surface.append(x_surface)
            xs_upper.append(x_upper)
            xs_wind.append(x_wind)
            ys.append(y)
        return np.array(xs_surface), np.array(xs_upper), np.array(xs_wind), np.array(ys)

    def __len__(self):
        if self.split == 'train':
            return self.train_size
        else:
            return len(self.X_wind) - self.train_size

    def __getitem__(self, idx):
        if self.split == 'test':
            idx += self.train_size
        sequence_surface = torch.tensor(self.X_surface[idx], dtype=torch.float32)
        sequence_upper = torch.tensor(self.X_upper[idx], dtype=torch.float32)
        sequence_wind = torch.tensor(self.X_wind[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32).squeeze()  # Squeeze in case of single column
        return (sequence_surface, sequence_upper, sequence_wind), label


# Example usage
csv_file = 'wind_speed_data.csv'
input_surface_file = 'input_surface.npy'
input_upper_file = 'input_upper.npy'
seq_length = 10

dataset = WindTurbineDataset(csv_file, input_surface_file, input_upper_file, seq_length, split='train')
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

test_dataset = WindTurbineDataset(csv_file, input_surface_file, input_upper_file, seq_length, split='test')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
