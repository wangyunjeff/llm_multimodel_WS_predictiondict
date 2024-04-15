import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load data
data_path = 'H:\\Code_F\\llm_multimodel_WS_predictiondict\\data\\la-haute-borne-data-2013-2016_new-3columns.csv'
data = pd.read_csv(data_path, index_col=0, parse_dates=True)
data = data.iloc[:, 1:2]  # Assuming this is the target column
data = data.interpolate()  # Handling missing values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences and corresponding targets with multiple steps ahead
def create_sequences(data, seq_length, pred_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - pred_length + 1):
        x = data[i:(i + seq_length), :]
        y = data[(i + seq_length):(i + seq_length + pred_length), 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 20
pred_length = 1  # Prediction length is 2
X, y = create_sequences(scaled_data, seq_length, pred_length)

# Chronological train-test split
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Naive forecast: Use the last observed value from each sequence for the predictions
naive_forecasts = np.array([X_test[i, -1, 0] for i in range(len(X_test))])
naive_forecasts = np.repeat(naive_forecasts, pred_length).reshape(-1, pred_length)

# Calculate MSE for the naive forecast
naive_mse = mean_squared_error(y_test.flatten(), naive_forecasts.flatten())
print(f'Naive forecast MSE: {naive_mse}')

# Plotting the results for the test set
plt.figure(figsize=(15, 6))
plt.plot(y_test.flatten(), label='Actual Data')
plt.plot(naive_forecasts.flatten(), label='Naive Forecast', linestyle='--')
plt.title('Naive Forecast vs Actual Data Over Entire Test Set')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Values')
plt.legend()
plt.show()
