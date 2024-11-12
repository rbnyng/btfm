# collection of backbones to be used in the model
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet18

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
        # mask = (x.sum(dim=-1) == 0)  # Create mask where all bands are 0

        # debug TODO: cannot do here, need to write a new script
        # if mask.sum() > 0:
        #     print(mask.shape)
        #     print("mask is not all zero!") 

        x = self.embedding(x)  # (batch_size, seq_length, latent_dim) -> (512, 16, 128)
        x = rearrange(x, 'b s d -> s b d')  # (16, 512, 128)
        x = self.pos_encoder(x)  # (16, 512, 128)
        
        # Pass mask to transformer_encoder
        # x = self.transformer_encoder(x, src_key_padding_mask=mask)  # (16, 512, 128)
        x = self.transformer_encoder(x)  # (16, 512, 128)
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
    
class SpatioTemporalResNet(nn.Module):
    def __init__(self, latent_dim, input_channels=11, pretrained=True):
        super(SpatioTemporalResNet, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        # Modify the first convolutional layer to adapt to 11 input channels
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Modify the last layer to output the specified latent_dim
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, latent_dim)
        self.temporal_aggregation = nn.Conv1d(in_channels=latent_dim, out_channels=latent_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: (batch_size, timestep, width, height, band)
        batch_size, timestep, width, height, band = x.shape
        # Merge the time dimension and batch dimension
        x = x.reshape(batch_size * timestep, band, width, height)
        # Extract spatial features
        features = self.resnet(x)  # Output shape: (batch_size * timestep, latent_dim)
        # Restore the time dimension
        features = features.view(batch_size, timestep, -1)  # Shape: (batch_size, timestep, latent_dim)
        features = features.permute(0, 2, 1)  # Convert to (batch_size, latent_dim, timestep)
        # 1D convolution to aggregate temporal features
        aggregated_features = self.temporal_aggregation(features)  # Output shape: (batch_size, latent_dim, timestep)
        # Take the output of the last time step
        latent_representation = aggregated_features[:, :, -1]  # Shape: (batch_size, latent_dim)
        return latent_representation
    
    
class TransformerEncoderWithMask(nn.Module):
    def __init__(self, input_dim=10, embed_dim=64, num_heads=8, hidden_dim=256, num_layers=6, latent_dim=128, dropout=0.2):
        super(TransformerEncoderWithMask, self).__init__()
        self.latent_dim = latent_dim
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # self.embedding = nn.Sequential(
            # nn.Linear(input_dim, embed_dim),
            # nn.BatchNorm1d(96)  # BatchNorm across the sequence length
        # )
        
        # 位置编码，长度为 96 的序列
        self.position_encoding = nn.Parameter(torch.randn(1, 96, embed_dim))
        
        # Transformer Encoder 层
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim, 
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # 注意力权重生成层，使用多层感知机结构
        # self.attention_weights = nn.Sequential(
            # nn.Linear(embed_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, 1)
        # )
        
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(96),  # BatchNorm across sequence length
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出层，包含 LayerNorm 和 Dropout，降维到 latent_dim
        # self.output_layer = nn.Sequential(
            # nn.LayerNorm(embed_dim),
            # nn.Dropout(dropout),
            # nn.Linear(embed_dim, latent_dim)
        # )
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),  # Keep LayerNorm here as it's standard in Transformers
            nn.Dropout(dropout),
            nn.Linear(embed_dim, latent_dim),
            nn.BatchNorm1d(latent_dim)  # Final BatchNorm on the output features
        )
    
    def forward(self, x, mask=None):
        # 输入嵌入
        x = self.embedding(x)  # (batch_size, 96, embed_dim)
        
        # Clip input values to prevent extremes
        x = torch.clamp(x, min=-100, max=100)
        
        # 添加位置编码
        x = x + self.position_encoding
        
        # 处理遮掩 mask
        if mask is not None:
            attention_mask = mask == 0  # PyTorch Transformer expects padding mask as `True` for masked positions
        else:
            attention_mask = None
        
        # Transformer Encoder 前向传播
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_key_padding_mask=attention_mask)
        
        # 自注意力池化，对时间维度加权求和
        weights = self.attention_weights(x)  # (batch_size, 96, 1)
        
        # 将 mask 应用于权重，将无效位置设为极小值以避免参与 softmax
        # if mask is not None:
        #     weights = weights.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
        # 对有效位置进行 softmax，以保证无效位置的权重为零
        weights = torch.softmax(weights, dim=1)  # (batch_size, 96, 1)
        
        # 对时间维度加权求和
        x = (x * weights).sum(dim=1)  # (batch_size, embed_dim)
        
        # 映射到 latent_dim
        output = self.output_layer(x)  # (batch_size, latent_dim)
        
        return output


class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        # 加载预训练的ResNet-18模型
        self.resnet = resnet18(pretrained=True)
        
        # 修改第一层以适应输入形状 (1, 96, 10)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.latent_dim = 128
        
        # 修改最后一层输出
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.latent_dim)  # 输出大小为128

    def forward(self, x):
        # 输入需要是 (batch_size, 1, 96, 10)，但是目前x是 (batch_size, 96, 10)
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x

# class TransformerEncoderWithMask(nn.Module):
#     def __init__(self, band_dim=11, time_steps=99, embed_dim=128, num_heads=4, num_layers=2):
#         super(TransformerEncoderWithMask, self).__init__()
#         self.band_dim = band_dim
#         self.time_steps = time_steps
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.num_layers = num_layers
        
#         # 1. 定义波段嵌入层
#         self.band_embedding = nn.Embedding(band_dim, embed_dim)
        
#         # 2. 定义时间嵌入层
#         self.time_embedding = nn.Embedding(time_steps, embed_dim)
        
#         # 3. 定义Transformer的编码层
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    
#     def forward(self, x, mask):
#         """
#         x: 输入数据，形状为 (batch_size, time_steps, band_dim)
#         mask: 时间维度掩码，形状为 (batch_size, time_steps)
#         """
#         batch_size, time_steps, band_dim = x.size()

#         # 1. 生成波段嵌入 (band_embedding)
#         # 这里将band_indices的形状设置为(batch_size, time_steps, band_dim) 而不是展平
#         band_indices = torch.arange(band_dim).unsqueeze(0).unsqueeze(0).repeat(batch_size, time_steps, 1).to(x.device)
#         band_embed = self.band_embedding(band_indices)  # (batch_size, time_steps, band_dim, embed_dim)
        
#         # 2. 生成时间嵌入 (time_embedding)
#         # time_indices 的形状为 (batch_size, time_steps, 1)
#         time_indices = torch.arange(time_steps).unsqueeze(0).repeat(batch_size, 1).to(x.device)
#         time_embed = self.time_embedding(time_indices).unsqueeze(2)  # (batch_size, time_steps, 1, embed_dim)
        
#         # 3. 将输入数据 x 扩展并加上嵌入信息
#         x = x.unsqueeze(-1)  # 将 x 扩展为 (batch_size, time_steps, band_dim, 1)
#         x_embed = x * band_embed + time_embed  # (batch_size, time_steps, band_dim, embed_dim)
        
#         # 4. 将数据 reshape 为 (batch_size, time_steps, embed_dim)
#         # 对 band_dim 进行均值池化，以将维度降到 Transformer 所需的 (batch_size, time_steps, embed_dim)
#         x_embed = x_embed.mean(dim=2)  # (batch_size, time_steps, embed_dim)
        
#         # 5. 掩码处理：调整掩码形状以适应Transformer的输入
#         attention_mask = mask == 0  # PyTorch Transformer expects `True` for masked positions
        
#         # 6. 使用Transformer进行编码
#         output = self.transformer_encoder(x_embed, src_key_padding_mask=attention_mask)  # (batch_size, time_steps, embed_dim)
        
#         # 7. 对编码后的特征进行全局池化 (这里选择平均池化，也可以使用其他方式)
#         output = output.mean(dim=1)  # (batch_size, embed_dim)
        
#         return output