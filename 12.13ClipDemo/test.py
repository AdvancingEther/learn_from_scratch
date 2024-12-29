import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from main import CLIP
import argparse
import os

@torch.no_grad()
def evaluate(model, test_loader):
    model.eval()
    correct_i2t = 0
    correct_t2i = 0
    total = 0
    
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        batch_size = images.shape[0]
        
        image_features = model.image_encoder(images)
        text_labels = torch.arange(10).cuda()
        text_features = model.text_encoder(text_labels)
        
        similarity = torch.matmul(image_features, text_features.t()) / model.temperature
        
        # 图像到文本的准确率
        predicted_labels = similarity.argmax(dim=1)
        correct_i2t += (predicted_labels == labels).sum().item()
        
        # 文本到图像的准确率
        label_similarities = similarity[torch.arange(batch_size), labels]
        for i, label in enumerate(labels):
            text_sim = similarity[:, label]
            if torch.argmax(text_sim) == i:
                correct_t2i += 1
        
        total += batch_size
    
    return {
        'i2t_accuracy': correct_i2t / total * 100,
        't2i_accuracy': correct_t2i / total * 100
    }

def visualize_predictions(model, test_loader, num_samples=5):
    """可视化图像到文本和文本到图像的预测结果"""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples].cuda(), labels[:num_samples].cuda()
    
    with torch.no_grad():
        # 图像到文本的预测
        image_features = model.image_encoder(images)
        text_labels = torch.arange(10).cuda()
        text_features = model.text_encoder(text_labels)
        i2t_similarity = torch.matmul(image_features, text_features.t()) / model.temperature
        i2t_predicted = i2t_similarity.argmax(dim=1)
        
        # 文本到图像的预测
        # 为了演示，我们使用原始标签作为查询文本
        query_text_features = model.text_encoder(labels)
        # 获取一批图像用于检索
        all_images, _ = next(iter(test_loader))
        all_images = all_images.cuda()
        all_image_features = model.image_encoder(all_images)
        t2i_similarity = torch.matmul(query_text_features, all_image_features.t()) / model.temperature
        t2i_predicted = t2i_similarity.argmax(dim=1)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, num_samples, figsize=(15, 6))
        plt.suptitle('上排：图像到文本的预测\n下排：文本到图像的检索结果')
        
        # 显示图像到文本的预测
        for i in range(num_samples):
            ax1[i].imshow(images[i].cpu().squeeze(), cmap='gray')
            ax1[i].axis('off')
            ax1[i].set_title(f'True: {labels[i].item()}\nPred: {i2t_predicted[i].item()}')
        
        # 显示文本到图像的预测
        for i in range(num_samples):
            retrieved_image = all_images[t2i_predicted[i]]
            ax2[i].imshow(retrieved_image.cpu().squeeze(), cmap='gray')
            ax2[i].axis('off')
            ax2[i].set_title(f'Query: {labels[i].item()}\nRetrieved')
        
        plt.tight_layout()
        plt.show()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试CLIP模型')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/clip_best.pth',
                      help='模型检查点路径 (默认: checkpoints/clip_best.pth)')
    parser.add_argument('--batch_size', type=int, default=10,
                      help='测试批次大小 (默认: 10)')
    parser.add_argument('--num_vis', type=int, default=10,
                      help='可视化样本数量 (默认: 10)')
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"模型文件未找到: {args.checkpoint}")
    
    # 加载测试数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 加载模型
    model = CLIP().cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"已加载模�� {args.checkpoint}")
    print(f"模型来自 Epoch {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    # 评估模型
    metrics = evaluate(model, test_loader)
    print("\n测试结果:")
    print(f"图像到文本准确率: {metrics['i2t_accuracy']:.2f}%")
    print(f"文本到图像准确率: {metrics['t2i_accuracy']:.2f}%")
    
    # 可视化预测结果
    print("\n正在生成可视化结果...")
    visualize_predictions(model, test_loader, num_samples=args.num_vis)

if __name__ == '__main__':
    main() 