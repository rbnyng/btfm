import torch
from torch import nn
import logging

class BTModel(nn.Module):
    def __init__(self, backbone, projector):
        super().__init__()

        self.backbone = backbone
        self.projector = projector

    def forward(self, x, week):
        return self.projector(self.backbone(x))

class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size, lambda_coeff=5e-3):
        super().__init__()

        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0) # torch.Size([256, 128])
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return (on_diag + self.lambda_coeff * off_diag), on_diag, self.lambda_coeff * off_diag

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.projection_head(x)
    
class EncoderModel(nn.Module):
    def __init__(self, backbone, time_dimension):
        super(EncoderModel, self).__init__()
        self.backbone = backbone
        self.time_dim = time_dimension
        self.time_embedding = None if time_dimension == 0 else nn.Embedding(53, time_dimension)

    def forward(self, x, time):
        if self.time_embedding is not None:
            time = self.time_embedding(time)
            if x.size(0) != time.size(0):
                raise ValueError("Batch size of x and time should match")
            # concatenate the two tensors along the last dimension
            x = torch.cat([x, time], dim=-1)
        else:
            if time.size(0) != x.size(0):
                raise ValueError("Batch size of x and time should match")
        features = self.backbone(x)
        return features