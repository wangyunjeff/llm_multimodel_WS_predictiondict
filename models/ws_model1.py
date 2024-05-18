import torch
import torch.nn as nn

class WindSpeedPredictionModel(nn.Module):
    def __init__(self):
        super(WindSpeedPredictionModel, self).__init__()

        # 2D CNN for Surface Data
        self.surface_cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # 3D CNN for Upper-Level Data
        self.upper_cnn = nn.Sequential(
            nn.Conv3d(5, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Flatten()
        )

        # LSTM for integrating and making predictions
        self.lstm = nn.LSTM(input_size=513, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, surface_data, upper_data, wind_speed_data):
        # Handle the extra time dimension by reshaping
        batch_size, seq_len = surface_data.size(0), surface_data.size(1)
        surface_data = surface_data.view(batch_size * seq_len, 4, 11, 11)
        upper_data = upper_data.view(batch_size * seq_len, 5, 13, 11, 11)
        wind_speed_data = wind_speed_data.view(batch_size * seq_len, -1)  # Flatten or directly use

        # Extract features
        surface_features = self.surface_cnn(surface_data)
        upper_features = self.upper_cnn(upper_data)

        # Concatenate features and reshape to include the sequence dimension again
        combined_features = torch.cat((surface_features, upper_features, wind_speed_data), dim=1)
        combined_features = combined_features.view(batch_size, seq_len, -1)

        # LSTM processing
        lstm_out, _ = self.lstm(combined_features)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step

        # Final prediction
        output = self.fc(lstm_out)
        return output


if __name__ == '__main__':
    # Function to generate dummy data
    def generate_dummy_data(batch_size, len_seq):
        # Create dummy data for surface variables with shape (batch_size, len_seq, 4, 11, 11)
        surface_data = torch.randn(batch_size, len_seq, 4, 11, 11)

        # Create dummy data for upper-level variables with shape (batch_size, len_seq, 5, 13, 11, 11)
        upper_data = torch.randn(batch_size, len_seq, 5, 13, 11, 11)

        # Create dummy data for wind speed time series with shape (batch_size, len_seq, 1)
        wind_speed_data = torch.randn(batch_size, len_seq, 1)

        return surface_data, upper_data, wind_speed_data


    # Initialize the model
    model = WindSpeedPredictionModel()

    # Generate dummy data
    batch_size = 5  # Number of samples in a batch
    len_seq = 10  # Length of the sequence (number of time steps)
    surface_data, upper_data, wind_speed_data = generate_dummy_data(batch_size, len_seq)

    # Simulate a forward pass
    with torch.no_grad():  # Disables gradient calculation to save memory and computations
        prediction = model(surface_data, upper_data, wind_speed_data)

    # Print the output
    print("Predicted Wind Speeds:")
    print(prediction)
