import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Load data
data_path = 'H:\\Code_F\\llm_multimodel_WS_predictiondict\\data\\la-haute-borne-data-2013-2016_new-3columns.csv'
data = pd.read_csv(data_path, index_col=0, parse_dates=True)
data = data.iloc[:, 1:2]
# Normalize and interpolate the data
data = data.interpolate()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        # x = data[i:(i + seq_length), :]
        x = data[i:(i + seq_length), :]
        y = data[i + seq_length, 0]  # Assuming column 1 is the target variable
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 20
X, y = create_sequences(scaled_data, seq_length)

# Chronological train-test split
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Create Tensor datasets
train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

# Define a batch size
batch_size = 64

# Create DataLoaders
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# LSTM model definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# Model instantiation
model = LSTM(input_size=1, hidden_layer_size=100, output_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
epochs = 500
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for seq, labels in train_loader:
        seq, labels = seq.to(device), labels.to(device)

        optimizer.zero_grad()
        y_pred = model(seq)

        loss = loss_function(y_pred.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} average loss: {avg_loss}')
    if epoch%20 ==0:
        # Plotting predictions after training
        model.eval()
        actual = []
        predicted = []

        with torch.no_grad():
            for seq, labels in test_loader:
                seq, labels = seq.to(device), labels.to(device)
                y_pred_test = model(seq)
                actual.extend(labels.cpu().numpy())
                predicted.extend(y_pred_test.squeeze().cpu().numpy())

        plt.figure(figsize=(30, 6))
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.title('Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Target Variable')
        plt.legend()
        plt.show()
