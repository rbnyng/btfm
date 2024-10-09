import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

# 1. 数据加载
representation_path = '/maps/zf281/btfm-data-preparation/test/MGRS-12TYN/representation_map.npy'
labels_path = '/maps/zf281/btfm-data-preparation/test/MGRS-12TYN/labels.npy'

representations = np.load(representation_path)
labels = np.load(labels_path) # values are [0, 10, 20, ..., 100]
# scale them to [0,1,2,3,...,10]
labels = (labels // 10).astype(np.int64)

# 2. 自定义Dataset类
class LandClassificationDataset(Dataset):
    def __init__(self, representations, labels):
        self.representations = representations
        self.labels = labels

    def __len__(self):
        return self.representations.shape[0] * self.representations.shape[1]

    def __getitem__(self, idx):
        row = idx // self.representations.shape[1]
        col = idx % self.representations.shape[1]
        return (
            torch.tensor(self.representations[row, col], dtype=torch.float32),
            torch.tensor(self.labels[row, col], dtype=torch.long)
        )

# 创建数据集对象
dataset = LandClassificationDataset(representations, labels)

# 3. 数据分割，按照8:2的比例划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# debug
train_dataset = dataset
val_dataset = dataset

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=12, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False, num_workers=12, pin_memory=True)

# 4. 定义模型
# class ClassificationHead(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super(ClassificationHead, self).__init__()
#         self.fc = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         return self.fc(x)

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # Layer 1
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        # Output layer (no activation here as we'll use softmax in the loss function)
        x = self.fc2(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
        if self.alpha is None:
            self.alpha = torch.tensor(1.0)

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# 设备检测：如果有CUDA可用，则使用它
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 初始化模型、损失函数和优化器
input_dim = representations.shape[-1]
num_classes = 11  # 标签的值为[0, 10, 20, ..., 100]，共有11类
model = ClassificationHead(input_dim, num_classes).to(device)  # 将模型移动到设备上
criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数移动到设备上
# criterion = FocalLoss().to(device)  # 将损失函数移动到设备上
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 定义学习率调度函数
decay_rate = 0.7  # 每次衰减到原来的70%
decay_steps = 500  # 每200步衰减一次

# Lambda函数，用于定义学习率的衰减策略
lambda_lr = lambda step: decay_rate ** (step // decay_steps)

# 设置学习率调度器
# scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

# 训练和验证循环
num_epochs = 50
log_interval = 5  # 每隔5个步骤打印日志
step = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_preds = []
    train_targets = []

    for i, (representations, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True)):
        # 将数据移动到设备上
        representations, labels = representations.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(representations)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        running_loss += loss.item()
        train_preds.extend(outputs.argmax(dim=1).cpu().numpy())  # 将预测结果移动到CPU以便计算
        train_targets.extend(labels.cpu().numpy())
        
        # if step == 1000:
        #     print("debug")

        # 每隔5个step打印一次训练日志
        if (step + 1) % log_interval == 0:
            train_loss = running_loss / log_interval
            train_accuracy = accuracy_score(train_targets, train_preds)
            f1 = f1_score(train_targets, train_preds, average='weighted')
            tqdm.write(f"Step [{step + 1}/{len(train_loader)}] - "
                       f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, F1: {f1:.4f}, lr: {optimizer.param_groups[0]['lr']}")
            running_loss = 0.0  # 重置running_loss以便下一个interval
        
        step += 1

    # 完成一个epoch后，进行验证并打印结果
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for representations, labels in val_loader:
            # 将验证数据移动到设备上
            representations, labels = representations.to(device), labels.to(device)

            outputs = model(representations)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())  # 将预测结果移动到CPU以便计算
            val_targets.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(val_targets, val_preds)
    f1 = f1_score(val_targets, val_preds, average='weighted')

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, F1: {f1:.4f}")