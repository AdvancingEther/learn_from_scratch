import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler
import os
import random

# 自定义采样器，确保每个batch中的样本类别不重复
class UniqueLabelSampler(Sampler):
    def __init__(self, labels, batch_size=10):
        self.labels = labels
        self.batch_size = batch_size
        self.labels_set = list(set(labels.numpy()))
        self.label_to_indices = {
            label: torch.where(labels == label)[0] 
            for label in self.labels_set
        }
        
    def __iter__(self):
        indices = []
        while len(indices) < len(self.labels):
            # 随机选择batch_size个不同的类别
            labels_subset = random.sample(self.labels_set, self.batch_size)
            # 对于每个类别随机选择一个样本
            for label in labels_subset:
                index = random.choice(self.label_to_indices[label]).item()
                indices.append(index)
        return iter(indices)
    
    def __len__(self):
        return len(self.labels)


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(128 * 3 * 3, embedding_dim)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x / x.norm(dim=-1, keepdim=True)

# 文本编码器 - 简单Embedding
class TextEncoder(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=512):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        return x / x.norm(dim=-1, keepdim=True)

# CLIP模型
class CLIP(nn.Module):
    def __init__(self, embedding_dim=512, temperature=0.07):
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.text_encoder = TextEncoder(num_classes=10, embedding_dim=embedding_dim)
        self.temperature = temperature
        
    def forward(self, images, labels):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(labels)
        
        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        return logits

# 训练函数
def train_clip(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        logits = model(images, labels)
        
        # 计算图像到文本的损失 (image to text)
        loss_i2t = nn.CrossEntropyLoss()(logits, torch.arange(len(images)).cuda())
        
        # 计算文本到图像的损失 (text to image)
        # 转置logits矩阵来计算文本到图像的匹配
        loss_t2i = nn.CrossEntropyLoss()(logits.t(), torch.arange(len(images)).cuda())
        
        # 总损失为两个方向损失的平均值
        loss = (loss_i2t + loss_t2i) / 2
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, '
                  f'Loss: {loss.item():.4f} '
                  f'(I2T: {loss_i2t.item():.4f}, T2I: {loss_t2i.item():.4f})')
    
    return total_loss / num_batches

# 主函数
def main():
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # 使用自定义采样器
    sampler = UniqueLabelSampler(train_dataset.targets, batch_size=10)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=10,  # 修改为10，因为MNIST只有10个类别
        sampler=sampler,
        num_workers=4
    )
    
    # 初始化模型
    model = CLIP().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练模型
    num_epochs = 10
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        avg_loss = train_clip(model, train_loader, optimizer, epoch)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        
        torch.save(checkpoint, os.path.join(save_dir, 'clip_latest.pth'))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, os.path.join(save_dir, 'clip_best.pth'))
            print(f'Epoch {epoch}: 保存最佳模型，损失为 {best_loss:.4f}')

if __name__ == '__main__':
    main()
