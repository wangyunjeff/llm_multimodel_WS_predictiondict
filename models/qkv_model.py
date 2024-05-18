import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output


class WindSpeedPredictionModel(nn.Module):
    def __init__(self, surface_dim, upper_dim, time_series_dim, hidden_dim, num_heads):
        super(WindSpeedPredictionModel, self).__init__()
        self.surface_conv = nn.Conv2d(in_channels=surface_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.upper_conv = nn.Conv2d(in_channels=upper_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        self.time_series_proj = nn.Linear(time_series_dim, hidden_dim)
        self.surface_proj = nn.Linear(hidden_dim * 11 * 11, hidden_dim)
        self.upper_proj = nn.Linear(hidden_dim * 11 * 11, hidden_dim)

        self.cross_attention = CrossAttention(dim=hidden_dim, num_heads=num_heads)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, surface, upper, time_series):
        batch_size, seq_len, *surface_dims = surface.shape
        _, _, *upper_dims = upper.shape

        surface = surface.view(batch_size * seq_len, *surface_dims)
        upper = upper.view(batch_size * seq_len, upper_dims[0]*upper_dims[1], upper_dims[2], upper_dims[3])

        surface_features = self.surface_conv(surface).view(batch_size, seq_len, self.surface_conv.out_channels, 11, 11)
        upper_features = self.upper_conv(upper).view(batch_size, seq_len, self.upper_conv.out_channels, 11, 11)

        # Flatten surface and upper features
        surface_features_flat = surface_features.flatten(start_dim=2)  # (batch_size, seq_len, 7744)
        upper_features_flat = upper_features.flatten(start_dim=2)  # (batch_size, seq_len, 7744)

        # Project to hidden_dim
        surface_features_flat_proj = self.surface_proj(surface_features_flat)  # (batch_size, seq_len, hidden_dim)
        upper_features_flat_proj = self.upper_proj(upper_features_flat)  # (batch_size, seq_len, hidden_dim)
        time_series_proj = self.time_series_proj(time_series)  # (batch_size, seq_len, hidden_dim)

        cross_attn_output = self.cross_attention(query=time_series_proj, key=surface_features_flat_proj,
                                                 value=upper_features_flat_proj)
        output = self.fc(cross_attn_output)

        return output


# Example usage
batch_size = 8
surface_dim = (4, 11, 11)
upper_dim = (5, 13, 11, 11)
time_series_dim = 10
seq_len = 100
hidden_dim = 64
num_heads = 4

surface = torch.randn(batch_size, seq_len, *surface_dim)
upper = torch.randn(batch_size, seq_len, *upper_dim)
time_series = torch.randn(batch_size, seq_len, time_series_dim)

model = WindSpeedPredictionModel(surface_dim=surface_dim[0], upper_dim=upper_dim[0]*upper_dim[1], time_series_dim=time_series_dim,
                                 hidden_dim=hidden_dim, num_heads=num_heads)
output = model(surface, upper, time_series)

print(output.shape)  # Expected shape: (batch_size, seq_len)
