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


model_SGDM_01 = ResNet18().to(device)
model_SGDM_001 = ResNet18().to(device)
model_SGDM_0001 = ResNet18().to(device)

model_Adam_01 = ResNet18().to(device)
model_Adam_001= ResNet18().to(device)
model_Adam_0001 = ResNet18().to(device)

models = [model_SGDM_01, model_SGDM_001, model_SGDM_0001, model_Adam_01, model_Adam_001, model_Adam_0001]

# 定义优化器
optimizers = {
    'SGDM_01': optim.SGD(model_SGDM_01.parameters(), lr=0.1, momentum=0.9),
    'SGDM_001': optim.SGD(model_SGDM_001.parameters(), lr=0.01, momentum=0.9),
    'SGDM_0001': optim.SGD(model_SGDM_0001.parameters(), lr=0.001, momentum=0.9),

    'Adam_01': optim.Adam(model_Adam_01.parameters(), lr=0.1),
    'Adam_001': optim.Adam(model_Adam_001.parameters(), lr=0.01),
    'Adam_0001': optim.Adam(model_Adam_0001.parameters(), lr=0.001)
}

# TensorBoard的记录器

writer1 = SummaryWriter('./runs/lr/SGDM_01')
writer2 = SummaryWriter('./runs/lr/SGDM_001')
writer3 = SummaryWriter('./runs/lr/SGDM_0001')
writer4 = SummaryWriter('./runs/lr/Adam_01')
writer5 = SummaryWriter('./runs/lr/Adam_001')
writer6 = SummaryWriter('./runs/lr/Adam_0001')



# 训练函数
def train_model(optimizer, model, train_loader, test_loader, criterion, idx, epochs ):

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
            loss.backward()
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

        # 记录测试集上的损失、准确率、精确率、召回率到 TensorBoard（统一添加标签用于区分不同优化器）
        precision, recall, accuracy, f1 = evaluate_model(model, test_loader)
        if idx == 1:
            writer1.add_scalar('Loss/SGDM', avg_loss_test, epoch )
            writer1.add_scalar('Accuracy/SGDM', accuracy, epoch)
            writer1.add_scalar('Precision/SGDM', precision, epoch)
            writer1.add_scalar('Recall/SGDM', recall, epoch)
            writer1.add_scalar('F1/SGDM', f1, epoch)
        elif idx == 2:
            writer2.add_scalar('Loss/SGDM', avg_loss_test, epoch )
            writer2.add_scalar('Accuracy/SGDM', accuracy, epoch)
            writer2.add_scalar('Precision/SGDM', precision, epoch)
            writer2.add_scalar('Recall/SGDM', recall, epoch)
            writer2.add_scalar('F1/SGDM', f1, epoch)
        elif idx == 3:
            writer3.add_scalar('Loss/SGDM', avg_loss_test, epoch )
            writer3.add_scalar('Accuracy/SGDM', accuracy, epoch)
            writer3.add_scalar('Precision/SGDM', precision, epoch)
            writer3.add_scalar('Recall/SGDM', recall, epoch)
            writer3.add_scalar('F1/SGDM', f1, epoch)
        elif idx == 4:
            writer4.add_scalar('Loss/Adam', avg_loss_test, epoch )
            writer4.add_scalar('Accuracy/Adam', accuracy, epoch)
            writer4.add_scalar('Precision/Adam', precision, epoch)
            writer4.add_scalar('Recall/Adam', recall, epoch)
            writer4.add_scalar('F1/Adam', f1, epoch)
        elif idx == 5:
            writer5.add_scalar('Loss/Adam', avg_loss_test, epoch )
            writer5.add_scalar('Accuracy/Adam', accuracy, epoch)
            writer5.add_scalar('Precision/Adam', precision, epoch)
            writer5.add_scalar('Recall/Adam', recall, epoch)
            writer5.add_scalar('F1/Adam', f1, epoch)
        elif idx == 6:
            writer6.add_scalar('Loss/Adam', avg_loss_test, epoch )
            writer6.add_scalar('Accuracy/Adam', accuracy, epoch)
            writer6.add_scalar('Precision/Adam', precision, epoch)
            writer6.add_scalar('Recall/Adam', recall, epoch)
            writer6.add_scalar('F1/Adam', f1, epoch)

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
    trained_model  = train_model(optimizer, model, train_loader, test_loader, criterion,idx = idx+1, epochs = 100 )


# 关闭 TensorBoard
writer1.close()
writer2.close()
writer3.close()
writer4.close()
writer5.close()
writer6.close()
