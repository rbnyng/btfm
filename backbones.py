# collection of backbones to be used in the model
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# each input is 11 bands

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # Use register_buffer to ensure `pe` is not a parameter

    def forward(self, x):
        # x shape: (seq_length, batch_size, latent_dim)
        seq_len = x.size(0)
        # Add positional encoding
        x = x + self.pe[:seq_len, :].unsqueeze(1).to(x.device)  # Shape: (seq_length, 1, latent_dim)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, sample_size, band_size, time_dim, latent_dim, nhead=8, num_encoder_layers=3, dim_feedforward=256, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = band_size
        self.embedding = nn.Linear(self.input_dim, latent_dim)  # Map input_dim (11) -> latent_dim (128)
        self.pos_encoder = PositionalEncoding(latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_dim) -> (512, 16, 11)
        x = self.embedding(x)  # (batch_size, seq_length, latent_dim) -> (512, 16, 128)
        x = rearrange(x, 'b s d -> s b d')  # (16, 512, 128)
        x = self.pos_encoder(x)  # (16, 512, 128)
        x = self.transformer_encoder(x)  #(16, 512, 128)
        x = rearrange(x, 's b d -> b s d')  # (512, 16, 128)
        x = x.mean(dim=1)  # (512, 128)
        x = self.fc_out(x)  # (512, 128)
        return x


# simple mlp layer, bn, relu
class SimpleMLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleMLPBlock, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

# simple three layer MLP
class SimpleMLP(torch.nn.Module):
    def __init__(self, sample_size, band_size, time_dim, latent_dim, hidden_dim, num_layers):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(sample_size*(band_size+time_dim), hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.layers = nn.ModuleList([SimpleMLPBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.fc_last = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = rearrange(x, 'b s n -> b (s n)')
        x = F.relu(self.fc1(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.fc_last(x)
        return x

# input is (batch_size, time_steps, bands) this simple CNN does
# treats (time_steps, bands) as a 2D image with one channel and applies
# a simple CNN with batchnorm
class SimpleCNN(torch.nn.Module):
    def __init__(self, sample_size, band_size, time_dim, latent_dim):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*sample_size*(band_size+time_dim), latent_dim)

    def forward(self, x):
        x = rearrange(x, 'b s n -> b 1 s n')
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x