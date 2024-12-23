import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from model import *


# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强与加载
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪并进行填充
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
 
# CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


# 定义损失函数
criterion = nn.CrossEntropyLoss()

b_01 = ResNet18().to(device)
b_03 = ResNet18().to(device)
b_06 = ResNet18().to(device)
b_10 = ResNet18().to(device)

models = [b_01, b_03, b_06, b_10]
b_list = [0.01, 0.03, 0.06, 0.1]
# 定义优化器
optimizers = {
    'b_01': optim.Adam(b_01.parameters(), lr=0.001),
    'b_03': optim.Adam(b_03.parameters(), lr=0.001),
    'b_06': optim.Adam(b_06.parameters(), lr=0.001),
    'b_10': optim.Adam(b_10.parameters(), lr=0.001)
}

# TensorBoard的记录器
writer1 = SummaryWriter('./runs/loss_b/b_01')
writer2 = SummaryWriter('./runs/loss_b/b_03')
writer3 = SummaryWriter('./runs/loss_b/b_06')
writer4 = SummaryWriter('./runs/loss_b/b_10')


# 训练函数
def train_model(optimizer, model, train_loader, test_loader, criterion, idx, epochs, b ):
    
    for epoch in range(epochs):
        model.train()
        epoch = epoch + 1
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)
            flood = (loss - b).abs() + b
            flood.backward()

            optimizer.step()

            running_loss += loss.item()

        # 每个epoch记录一次平均损失
        avg_loss_train = running_loss / len(train_loader)

        running_loss = 0.0
        model.eval()
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        avg_loss_test = running_loss / len(test_loader)

        # 记录训练集上的损失、准确率、精确率、召回率到 TensorBoard（统一添加标签用于区分不同优化器）
        precision, recall, accuracy, f1 = evaluate_model(model, train_loader)
        if idx == 1:
            writer1.add_scalar('Loss/train', avg_loss_train, epoch )
            writer1.add_scalar('Accuracy/train', accuracy, epoch)
            writer1.add_scalar('Precision/train', precision, epoch)
            writer1.add_scalar('Recall/train', recall, epoch)
            writer1.add_scalar('F1/train', f1, epoch)
        if idx == 2:
            writer2.add_scalar('Loss/train', avg_loss_train, epoch )
            writer2.add_scalar('Accuracy/train', accuracy, epoch)
            writer2.add_scalar('Precision/train', precision, epoch)
            writer2.add_scalar('Recall/train', recall, epoch)
            writer2.add_scalar('F1/train', f1, epoch)
        if idx == 3:
            writer3.add_scalar('Loss/train', avg_loss_train, epoch )
            writer3.add_scalar('Accuracy/train', accuracy, epoch)
            writer3.add_scalar('Precision/train', precision, epoch)
            writer3.add_scalar('Recall/train', recall, epoch)
            writer3.add_scalar('F1/train', f1, epoch)
        if idx == 4:
            writer4.add_scalar('Loss/train', avg_loss_train, epoch )
            writer4.add_scalar('Accuracy/train', accuracy, epoch)
            writer4.add_scalar('Precision/train', precision, epoch)
            writer4.add_scalar('Recall/train', recall, epoch)
            writer4.add_scalar('F1/train', f1, epoch)

        # 记录测试集上的损失、准确率、精确率、召回率到 TensorBoard（统一添加标签用于区分不同优化器）
        precision, recall, accuracy, f1 = evaluate_model(model, test_loader)
        if idx == 1:
            writer1.add_scalar('Loss/test', avg_loss_test, epoch )
            writer1.add_scalar('Accuracy/test', accuracy, epoch)
            writer1.add_scalar('Precision/test', precision, epoch)
            writer1.add_scalar('Recall/test', recall, epoch)
            writer1.add_scalar('F1/test', f1, epoch)
        if idx == 2:
            writer2.add_scalar('Loss/test', avg_loss_test, epoch )
            writer2.add_scalar('Accuracy/test', accuracy, epoch)
            writer2.add_scalar('Precision/test', precision, epoch)
            writer2.add_scalar('Recall/test', recall, epoch)
            writer2.add_scalar('F1/test', f1, epoch)
        if idx == 3:
            writer3.add_scalar('Loss/test', avg_loss_test, epoch )
            writer3.add_scalar('Accuracy/test', accuracy, epoch)
            writer3.add_scalar('Precision/test', precision, epoch)
            writer3.add_scalar('Recall/test', recall, epoch)
            writer3.add_scalar('F1/test', f1, epoch)
        if idx == 4:
            writer4.add_scalar('Loss/test', avg_loss_test, epoch )
            writer4.add_scalar('Accuracy/test', accuracy, epoch)
            writer4.add_scalar('Precision/test', precision, epoch)
            writer4.add_scalar('Recall/test', recall, epoch)
            writer4.add_scalar('F1/test', f1, epoch)

        print('epoch:', epoch , '---Loss: ', avg_loss_train, '---Accuracy:', accuracy)

    return model

# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 计算准确率，精确率，召回率
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return precision, recall, accuracy, f1

# 训练并评估模型，分别使用5种优化器
for idx, (opt_name, optimizer) in enumerate(optimizers.items()):
    print(f"Training with {opt_name} optimizer...")
    model = models[idx]  # 为每个优化器训练一个模型
    b_number = b_list[idx]
    trained_model = train_model(optimizer, model, train_loader, test_loader, criterion,idx = idx+1, epochs=100, b = b_number )


# 关闭 TensorBoard
writer1.close()
writer2.close()
writer3.close()