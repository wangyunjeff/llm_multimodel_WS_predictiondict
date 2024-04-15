import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
# Load data
data_path = 'H:\\Code_F\\llm_multimodel_WS_predictiondict\\data\\la-haute-borne-data-2013-2016_new-3columns.csv'
data = pd.read_csv(data_path, index_col=0, parse_dates=True)

# Normalize and interpolate the data
data = data.interpolate()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :]
        y = data[i + seq_length, 1]  # Assuming column 1 is the target variable
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10
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
model = LSTM(input_size=3, hidden_layer_size=100, output_size=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming timeseries is the original target variable (column 1 in your case)
timeseries = data.iloc[:, 1].values  # Update the column index if needed

# Length of the sequence used for prediction
lookback = seq_length

# Prepare the plot arrays
train_plot = np.empty_like(timeseries)
train_plot[:] = np.nan
test_plot = np.empty_like(timeseries)
test_plot[:] = np.nan




# Training the model
epochs = 100
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

    # Generate predictions and prepare the plotting data
    with torch.no_grad():
        model.eval()

        # Generate predictions for the training data
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        train_pred = model(X_train_tensor)[:, 0].cpu().numpy()
        train_plot[lookback:train_size + lookback] = train_pred

        # Generate predictions for the testing data
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        test_pred = model(X_test_tensor)[:, 0].cpu().numpy()
        test_plot[train_size + lookback:] = test_pred

    # Plotting
    plt.figure(figsize=(50, 6))
    plt.plot(timeseries, color='blue', label='Actual')
    plt.plot(train_plot, color='red', label='Train Predictions')
    plt.plot(test_plot, color='green', label='Test Predictions')
    plt.title('Time Series Prediction')
    plt.xlabel('Time')
    plt.ylabel('Target Variable')
    plt.legend()
    plt.show()





