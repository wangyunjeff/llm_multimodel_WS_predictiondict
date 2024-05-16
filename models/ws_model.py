import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        # query: (batch_size, seq_len, dim)
        # key, value: (batch_size, seq_len, dim)
        attn_output, _ = self.attention(query, key, value)
        return attn_output


class WindSpeedPredictionModel(nn.Module):
    def __init__(self, surface_dim, upper_dim, time_series_dim, hidden_dim, num_heads):
        super(WindSpeedPredictionModel, self).__init__()
        self.surface_conv = nn.Conv2d(in_channels=surface_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.upper_conv = nn.Conv3d(in_channels=upper_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

        self.time_series_proj = nn.Linear(time_series_dim, hidden_dim)  # Project time_series to hidden_dim
        self.upper_proj = nn.Linear(hidden_dim * 13, hidden_dim)  # Project upper_features_flat to hidden_dim

        self.cross_attention = CrossAttention(dim=hidden_dim, num_heads=num_heads)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, surface, upper, time_series):
        # surface: (batch_size, 4, 11, 11)
        # upper: (batch_size, 5, 13, 11, 11)
        # time_series: (batch_size, seq_len, time_series_dim)

        surface_features = self.surface_conv(surface)  # (batch_size, hidden_dim, 11, 11)
        upper_features = self.upper_conv(upper)  # (batch_size, hidden_dim, 13, 11, 11)

        surface_features_flat = surface_features.view(surface_features.size(0), surface_features.size(1), -1).permute(0,
                                                                                                                      2,
                                                                                                                      1)  # (batch_size, 11*11, hidden_dim)
        upper_features_flat = upper_features.view(upper_features.size(0), upper_features.size(1),
                                                  upper_features.size(2), -1).permute(0, 3, 1, 2).contiguous().view(
            upper_features.size(0), -1, hidden_dim * 13)  # (batch_size, 11*11, hidden_dim * 13)

        upper_features_flat_proj = self.upper_proj(upper_features_flat)  # (batch_size, 11*11, hidden_dim)
        time_series_proj = self.time_series_proj(time_series)  # (batch_size, seq_len, hidden_dim)

        cross_attn_output = self.cross_attention(query=time_series_proj, key=surface_features_flat,
                                                 value=upper_features_flat_proj)  # (batch_size, seq_len, hidden_dim)

        output = self.fc(cross_attn_output)  # (batch_size, seq_len, 1)
        return output.squeeze(-1)


# Example usage
batch_size = 8
surface_dim = (4, 11, 11)
upper_dim = (5, 13, 11, 11)
time_series_dim = 10
seq_len = 100
hidden_dim = 64
num_heads = 4

surface = torch.randn(batch_size, *surface_dim)
upper = torch.randn(batch_size, *upper_dim)
time_series = torch.randn(batch_size, seq_len, time_series_dim)

model = WindSpeedPredictionModel(surface_dim=surface_dim[0], upper_dim=upper_dim[0], time_series_dim=time_series_dim,
                                 hidden_dim=hidden_dim, num_heads=num_heads)
output = model(surface, upper, time_series)

print(output.shape)  # (batch_size, seq_len)
