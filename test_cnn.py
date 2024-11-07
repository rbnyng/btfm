import torch
import torch.nn as nn
import torchvision.models as models

# 定义改进的ResNet-18模型
class ModifiedResNet18(nn.Module):
    def __init__(self):
        super(ModifiedResNet18, self).__init__()
        # 加载预训练的ResNet-18模型
        self.resnet = models.resnet18(pretrained=True)
        
        # 修改第一层以适应输入形状 (1, 96, 10)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # 修改最后一层输出
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 64)  # 输出大小为64

    def forward(self, x):
        # 输入需要是 (batch_size, 1, 96, 10)
        x = self.resnet(x)
        return x

# 创建模型实例
model = ModifiedResNet18()

# 计算参数数量
num_params = sum(p.numel() for p in model.parameters())

# 示例输入数据
input_data = torch.rand(100, 1, 96, 10)  # 100个样本，1个通道，96个时间步长，10个波段
output = model(input_data)

# 打印输出形状
print(output.shape)  # 应该是 (100, 64)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 假设二分类任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 示例标签
labels = torch.randint(0, 2, (100, 64)).float()  # 示例标签

# 训练循环示例
model.train()
for epoch in range(10):  # 训练10个epoch
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')
