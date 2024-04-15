import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data.dataloader import WindTurbineDataset

# Load data using the WindTurbineDataset class
seq_length = 20
path = 'H:\\Code_F\\llm_multimodel_WS_predictiondict\\data\\la-haute-borne-data-2013-2016_new-3columns.csv'
columns = ['Ws_avg']
train_dataset = WindTurbineDataset(path, seq_length, columns=columns, split='train')
test_dataset = WindTurbineDataset(path, seq_length, columns=columns, split='test')
# DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


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
# Training the model
epochs = 5000
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

    avg_train_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} average training loss: {avg_train_loss}')

    if epoch % 50 == 0:
        # Evaluate on test data
        model.eval()
        total_test_loss = 0
        actual = []
        predicted = []

        with torch.no_grad():
            for seq, labels in test_loader:
                seq, labels = seq.to(device), labels.to(device)
                y_pred_test = model(seq)
                actual.extend(labels.cpu().numpy())
                predicted.extend(y_pred_test.squeeze().cpu().numpy())

                # Compute test loss
                test_loss = loss_function(y_pred_test.squeeze(), labels)
                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f'Epoch {epoch} average test loss: {avg_test_loss}')

        # Plotting predictions after training
        plt.figure(figsize=(30, 6))
        plt.plot(actual, label='Actual')
        plt.plot(predicted, label='Predicted')
        plt.title('Actual vs Predicted')
        plt.xlabel('Time')
        plt.ylabel('Target Variable')
        plt.legend()
        plt.show()