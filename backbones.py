# collection of backbones to be used in the model
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import resnet18, resnet50
from ltae import LTAE2d
from einops import rearrange
import logging

DEBUG = True
# each input is 11 bands
def debug_tensor(name, tensor,debug=DEBUG):
    if not debug:
        return
    if tensor is not None:
        # max()只对于float tensor，因此需要先转换为float tensor
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        logging.info(f"{name} - max: {tensor.max().item()}, min: {tensor.min().item()}, mean: {tensor.mean().item()}")
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN!")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf!")
    else:
        logging.info(f"{name} is None")

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
    def __init__(self, band_num, latent_dim, nhead=8, num_encoder_layers=3, dim_feedforward=256, dropout=0.2, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = band_num
        self.embedding = nn.Linear(self.input_dim, latent_dim)  # Map input_dim (11) -> latent_dim (128)
        self.pos_encoder = PositionalEncoding(latent_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_length, latent_dim) -> (512, 25, 128)
        x = rearrange(x, 'b s d -> s b d')  # (25, 512, 128)
        x = self.pos_encoder(x)  # (25, 512, 128)
        
        x = self.transformer_encoder(x)  # (25, 512, 128)
        x = rearrange(x, 's b d -> b s d')  # (512, 25, 128)
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
    
class TemporalAttention(nn.Module):
    def __init__(self, latent_dim):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.latent_dim = latent_dim

    def forward(self, x):
        # x shape: (batch_size, latent_dim, timestep)
        x = x.permute(0, 2, 1)  # 转为 (batch_size, timestep, latent_dim)
        query = self.query(x)  # (batch_size, timestep, latent_dim)
        key = self.key(x)      # (batch_size, timestep, latent_dim)
        value = self.value(x)  # (batch_size, timestep, latent_dim)
        
        # 计算注意力得分
        attention_scores = torch.bmm(query, key.transpose(1, 2)) / (self.latent_dim ** 0.5)  # (batch_size, timestep, timestep)
        attention_weights = self.softmax(attention_scores)  # (batch_size, timestep, timestep)
        
        # 注意力加权和
        attended_features = torch.bmm(attention_weights, value)  # (batch_size, timestep, latent_dim)
        return attended_features.permute(0, 2, 1)  # 转回 (batch_size, latent_dim, timestep)


class SpatioTemporalResNet(nn.Module):
    def __init__(self, latent_dim, input_channels=10, pretrained=True):
        super(SpatioTemporalResNet, self).__init__()
        # self.resnet = resnet18(pretrained=pretrained)
        self.resnet = resnet50(pretrained=pretrained)
        # 修改第一层以适应更多输入通道
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 替换最后的全连接层为全局池化 + 映射到 latent_dim
        self.resnet.fc = nn.Identity()  # 移除全连接层
        # self.fc_out = nn.Linear(512, latent_dim)  # 映射到 latent_dim
        
        # 初始化 TemporalAttention
        self.temporal_attention = TemporalAttention(latent_dim)

    def forward(self, x):
        # x shape: (batch_size, timestep, height, width, band)
        batch_size, timestep, height, width, band = x.shape
        # 合并时间和批量维度
        x = rearrange(x, 'b t h w c -> (b t) c h w')
        # 提取空间特征
        spatial_features = self.resnet(x)  # (batch_size * timestep, 2048)
        # latent_features = self.fc_out(spatial_features)  # (batch_size * timestep, latent_dim)

        # 恢复时间维度
        latent_features = rearrange(spatial_features, '(b t) d -> b t d', b=batch_size)
        latent_features = latent_features.permute(0, 2, 1)  # 转为 (batch_size, latent_dim, timestep)

        # 使用 TemporalAttention 聚合时间步
        attended_features = self.temporal_attention(latent_features)  # (batch_size, latent_dim, timestep)

        # 聚合为最终的输出 (batch_size, latent_dim)
        # final_features, _ = attended_features.max(dim=2) # max
        final_features = attended_features.mean(dim=2) # mean
        return final_features

class SpatialResNet(nn.Module):
    def __init__(self, input_channels=10, pretrained=True):
        super(SpatialResNet, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)
        # 修改第一层以适应更多输入通道
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 替换最后的全连接层为全局池化 + 映射到 latent_dim
        self.resnet.fc = nn.Identity()  # 移除全连接层

    def forward(self, x):
        # 输入形状: (batch_size, channels, height, width)
        spatial_features = self.resnet(x)  # (batch_size, 2048, 1, 1)
        return spatial_features
    
class TransformerEncoderWithMask(nn.Module):
    def __init__(self, input_dim=10, embed_dim=128, num_heads=8, hidden_dim=256, num_layers=6, latent_dim=128, dropout=0.2):
        super(TransformerEncoderWithMask, self).__init__()
        self.latent_dim = latent_dim
        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, embed_dim)
        
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
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 输出层，包含 LayerNorm 和 Dropout，降维到 latent_dim
        self.output_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, latent_dim)
        )
    
    def forward(self, x, mask=None):
        # 输入嵌入
        x = self.embedding(x)  # (batch_size, 96, embed_dim)
        
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
        if mask is not None:
            weights = weights.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        
        # 对有效位置进行 softmax，以保证无效位置的权重为零
        weights = torch.softmax(weights, dim=1)  # (batch_size, 96, 1)
        
        # 对时间维度加权求和
        x = (x * weights).sum(dim=1)  # (batch_size, embed_dim)
        
        # 映射到 latent_dim
        output = self.output_layer(x)  # (batch_size, latent_dim)
        
        return output

class MultiScaleDownsampledTransformer(nn.Module):
    def __init__(self, input_dim=10, embed_dim=128, num_heads=8, hidden_dim=256, num_layers=6, latent_dim=128, num_scales=4, dropout=0.2):
        super(MultiScaleDownsampledTransformer, self).__init__()
        self.num_scales = num_scales
        self.scale_latent_dim = latent_dim // num_scales  # 每个尺度的子表示大小
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.position_encoding = nn.Parameter(torch.zeros(1, 96, embed_dim))
        torch.nn.init.trunc_normal_(self.position_encoding, std=0.02)  # 避免过大的初始值
        
        # Transformer Encoder，用于每个尺度的特征提取
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=num_layers
            ) for _ in range(num_scales)
        ])
        
        # 降维到每个尺度的 latent_dim
        self.output_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, self.scale_latent_dim)
            ) for _ in range(num_scales)
        ])
        
        # 自适应池化模块，用于时间维度下采样
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(output_size=96 // (2 ** i)) for i in range(num_scales)
        ])
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        
    def forward(self, x, mask=None):
        debug_tensor("x before embedding", x)
        x = self.embedding(x)  # (batch_size, 96, embed_dim)
        debug_tensor("x after embedding", x)
        x = x + self.position_encoding

        scale_representations = []
        for i in range(self.num_scales):
            debug_tensor(f"scale {i} x", x)
            logging.info(f"scale {i} x value: {x}")
            # 下采样时间维度
            x_downsampled = self.adaptive_pools[i]((x + 1e-8).transpose(1, 2)).transpose(1, 2)  # (batch_size, downsampled_length, embed_dim)
            # debug only, scale to 0-1
            x_downsampled = (x_downsampled - x_downsampled.min()) / (x_downsampled.max() - x_downsampled.min())
            # to (-1,1)
            x_downsampled = 2 * x_downsampled - 1
            
            debug_tensor("x_downsampled", x_downsampled)
            # 生成 mask 对应的下采样版本
            mask_downsampled = (
                nn.functional.adaptive_avg_pool1d(mask.unsqueeze(1).float(), output_size=x_downsampled.size(1)).squeeze(1).long()
                if mask is not None else None
            )
            debug_tensor("mask_downsampled", mask_downsampled)
            
            # 将mask转为bool类型
            mask_downsampled = mask_downsampled == 0 if mask_downsampled is not None else None
            logging.info(f"mask_downsampled value: {mask_downsampled}")
            debug_tensor("mask_downsampled", mask_downsampled)
            
            # Transformer Encoder 提取特征
            x_encoded = self.encoder_layers[i](x_downsampled, src_key_padding_mask=mask_downsampled)
            debug_tensor("x_encoded", x_encoded)
            # 降维到 scale_latent_dim
            scale_rep = self.output_layers[i](x_encoded.mean(dim=1))  # (batch_size, scale_latent_dim)
            debug_tensor("scale_rep", scale_rep)
            scale_representations.append(scale_rep)
            
        # 拼接所有尺度的表示，得到最终输出
        final_representation = torch.cat(scale_representations, dim=-1)  # (batch_size, latent_dim)
        return final_representation


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


class UTAE(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 128, 256, 512],
        decoder_widths=[32, 64, 128, 512],
        out_conv=[32, 20],
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=512,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
    ):
        """
        U-TAE architecture for spatio-temporal encoding of satellite image time series.
        Args:
            input_dim (int): Number of channels in the input images.
            encoder_widths (List[int]): List giving the number of channels of the successive encoder_widths of the convolutional encoder.
            This argument also defines the number of encoder_widths (i.e. the number of downsampling steps +1)
            in the architecture.
            The number of channels are given from top to bottom, i.e. from the highest to the lowest resolution.
            decoder_widths (List[int], optional): Same as encoder_widths but for the decoder. The order in which the number of
            channels should be given is also from top to bottom. If this argument is not specified the decoder
            will have the same configuration as the encoder.
            out_conv (List[int]): Number of channels of the successive convolutions for the
            str_conv_k (int): Kernel size of the strided up and down convolutions.
            str_conv_s (int): Stride of the strided up and down convolutions.
            str_conv_p (int): Padding of the strided up and down convolutions.
            agg_mode (str): Aggregation mode for the skip connections. Can either be:
                - att_group (default) : Attention weighted temporal average, using the same
                channel grouping strategy as in the LTAE. The attention masks are bilinearly
                resampled to the resolution of the skipped feature maps.
                - att_mean : Attention weighted temporal average,
                 using the average attention scores across heads for each date.
                - mean : Temporal average excluding padded dates.
            encoder_norm (str): Type of normalisation layer to use in the encoding branch. Can either be:
                - group : GroupNorm (default)
                - batch : BatchNorm
                - instance : InstanceNorm
            n_head (int): Number of heads in LTAE.
            d_model (int): Parameter of LTAE
            d_k (int): Key-Query space dimension
            encoder (bool): If true, the feature maps instead of the class scores are returned (default False)
            return_maps (bool): If true, the feature maps instead of the class scores are returned (default False)
            pad_value (float): Value used by the dataloader for temporal padding.
            padding_mode (str): Spatial padding strategy for convolutional layers (passed to nn.Conv2d).
        """
        super(UTAE, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.enc_dim = (
            decoder_widths[0] if decoder_widths is not None else encoder_widths[0]
        )
        self.stack_dim = (
            sum(decoder_widths) if decoder_widths is not None else sum(encoder_widths)
        )
        self.pad_value = pad_value
        self.encoder = encoder
        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        self.in_conv = ConvBlock(
            nkernels=[input_dim] + [encoder_widths[0], encoder_widths[0]],
            pad_value=pad_value,
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                pad_value=pad_value,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.temporal_encoder = LTAE2d(
            in_channels=encoder_widths[-1],
            d_model=d_model,
            n_head=n_head,
            mlp=[d_model, encoder_widths[-1]],
            return_att=True,
            d_k=d_k,
        )
        self.temporal_aggregator = Temporal_Aggregator(mode=agg_mode)
        self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode)
        self.gap = nn.AdaptiveAvgPool2d(1)  # 输出 1x1 的特征图

    def forward(self, input, batch_positions=None, return_att=False): # input: BxTxCxHxW
        pad_mask = (
            (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask
        out = self.in_conv.smart_forward(input) # BxTxCxHxW (chanel from 10 to 64)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1])
            feature_maps.append(out)
        # TEMPORAL ENCODER
        out, att = self.temporal_encoder(
            feature_maps[-1], batch_positions=batch_positions, pad_mask=pad_mask
        )
        
        pooled_out = self.gap(out).squeeze(-1).squeeze(-1)  # 最终形状为 [batch_size, 512]
        
        return pooled_out
        # SPATIAL DECODER
        # if self.return_maps:
        #     maps = [out]
        # for i in range(self.n_stages - 1):
        #     skip = self.temporal_aggregator(
        #         feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
        #     )
        #     out = self.up_blocks[i](out, skip)
        #     if self.return_maps:
        #         maps.append(out)

        # if self.encoder:
        #     return out, maps
        # else:
        #     out = self.out_conv(out)
        #     if return_att:
        #         return out, att
        #     if self.return_maps:
        #         return out, maps
        #     else:
        #         return out


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method the the block.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value=None):
        super(TemporallySharedBlock, self).__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, input):
        if len(input.shape) == 4:
            return self.forward(input)
        else:
            b, t, c, h, w = input.shape

            if self.pad_value is not None:
                dummy = torch.zeros(input.shape, device=input.device).float()
                self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape

            out = input.view(b * t, c, h, w)
            if self.pad_value is not None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
                if pad_mask.any():
                    temp = (
                        torch.ones(
                            self.out_shape, device=input.device, requires_grad=False
                        )
                        * self.pad_value
                    )
                    temp[~pad_mask] = self.forward(out[~pad_mask])
                    out = temp
                else:
                    out = self.forward(out)
            else:
                out = self.forward(out)
            _, c, h, w = out.shape
            out = out.view(b, t, c, h, w)
            return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvLayer, self).__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":
            nl = lambda num_feats: nn.GroupNorm(
                num_channels=num_feats,
                num_groups=n_groups,
            )
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        nkernels,
        pad_value=None,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super(ConvBlock, self).__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        pad_value=None,
        norm="batch",
        padding_mode="reflect",
    ):
        super(DownConvBlock, self).__init__(pad_value=pad_value)
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(nn.Module):
    def __init__(
        self, d_in, d_out, k, s, p, norm="batch", d_skip=None, padding_mode="reflect"
    ):
        super(UpConvBlock, self).__init__()
        d = d_out if d_skip is None else d_skip
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in, out_channels=d_out, kernel_size=k, stride=s, padding=p
            ),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class Temporal_Aggregator(nn.Module):
    def __init__(self, mode="mean"):
        super(Temporal_Aggregator, self).__init__()
        self.mode = mode

    def forward(self, x, pad_mask=None, attn_mask=None):
        if pad_mask is not None and pad_mask.any():
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                attn = attn * (~pad_mask).float()[:, :, None, None]
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                out = x * (~pad_mask).float()[:, :, None, None, None]
                out = out.sum(dim=1) / (~pad_mask).sum(dim=1)[:, None, None, None]
                return out
        else:
            if self.mode == "att_group":
                n_heads, b, t, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b, t, h, w)
                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=x.shape[-2:], mode="bilinear", align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                attn = attn.view(n_heads, b, t, *x.shape[-2:])
                out = torch.stack(x.chunk(n_heads, dim=2))  # hxBxTxC/hxHxW
                out = attn[:, :, :, None, :, :] * out
                out = out.sum(dim=2)  # sum on temporal dim -> hxBxC/hxHxW
                out = torch.cat([group for group in out], dim=1)  # -> BxCxHxW
                return out
            elif self.mode == "att_mean":
                attn = attn_mask.mean(dim=0)  # average over heads -> BxTxHxW
                attn = nn.Upsample(
                    size=x.shape[-2:], mode="bilinear", align_corners=False
                )(attn)
                out = (x * attn[:, :, None, :, :]).sum(dim=1)
                return out
            elif self.mode == "mean":
                return x.mean(dim=1)