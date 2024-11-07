import numpy as np
import matplotlib.pyplot as plt
# # np_file_path = "/maps/sj514/btfm/data/california/processed/MGRS-10SGD/masks.npy"
# # data = np.load(np_file_path)
# # print(data.shape)
# # print(data.dtype)
# # print(np.unique(data))
# # # print(np.sum(data == 0))

# import torch
# import torch.nn as nn

# class TransformerEncoderWithMask(nn.Module):
#     def __init__(self, band_dim=11, time_steps=96, embed_dim=128, num_heads=4, num_layers=2):
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

# # 测试数据
# batch_size = 256
# time_steps = 99
# band_dim = 11
# embed_dim = 128
# num_heads = 4
# num_layers = 2

# # 创建模型
# model = TransformerEncoderWithMask(band_dim=band_dim, time_steps=time_steps, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers).to('cuda')

# # 创建示例输入数据 (有些时间步的数据为缺失，用 0 填充表示)
# x = torch.randn(batch_size, time_steps, band_dim).to('cuda')
# mask = torch.randint(0, 2, (batch_size, time_steps)).to('cuda')  # 仅在时间维度上使用二值掩码

# # 模型输出
# output = model(x, mask)
# print(output.shape)

file_path = "/maps/zf281/pangaea-bench/data/PASTIS-HD/ANNOTATIONS/ParcelIDs_10000.npy"
data = np.load(file_path) #(46,10,128,128)
# single_image = data[0,0]
print(data.shape)
# 绘制并保存
# plt.imshow(single_image, cmap='gray')
# plt.savefig("single_image.png")