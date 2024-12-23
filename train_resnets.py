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

resnet18 = ResNet18().to(device)
resnet34 = ResNet34().to(device)
resnet50 = ResNet50().to(device)
resnet101 = ResNet101().to(device)
resnet152 = ResNet152().to(device)

models = [resnet18, resnet34, resnet50, resnet101, resnet152]

# 定义优化器
optimizers = {
    'Adam_resnet18': optim.Adam(resnet18.parameters(), lr=0.001),
    'Adam_resnet34': optim.Adam(resnet34.parameters(), lr=0.001),
    'Adam_resnet50': optim.Adam(resnet50.parameters(), lr=0.001),
    'Adam_resnet101': optim.Adam(resnet101.parameters(), lr=0.001),
    'Adam_resnet152': optim.Adam(resnet152.parameters(), lr=0.001),

}

# TensorBoard的记录器
writer1 = SummaryWriter('runs/models/resnet18')
writer2 = SummaryWriter('runs/models/resnet34')
writer3 = SummaryWriter('runs/models/resnet50')
writer4 = SummaryWriter('runs/models/resnet101')
writer5 = SummaryWriter('runs/models/resnet152')

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

        # 记录损失、准确率、精确率、召回率到 TensorBoard（统一添加标签用于区分不同优化器）
        precision, recall, accuracy, f1 = evaluate_model(model, train_loader)
        if idx == 1:
            writer1.add_scalar('Loss/train', avg_loss_train, epoch )
            writer1.add_scalar('Accuracy/train', accuracy, epoch)
            writer1.add_scalar('Precision/train', precision, epoch)
            writer1.add_scalar('Recall/train', recall, epoch)
            writer1.add_scalar('F1/train', f1, epoch)
        elif idx == 2:
            writer2.add_scalar('Loss/train', avg_loss_train, epoch )
            writer2.add_scalar('Accuracy/train', accuracy, epoch)
            writer2.add_scalar('Precision/train', precision, epoch)
            writer2.add_scalar('Recall/train', recall, epoch)
            writer2.add_scalar('F1/train', f1, epoch)
        elif idx == 3:
            writer3.add_scalar('Loss/train', avg_loss_train, epoch )
            writer3.add_scalar('Accuracy/train', accuracy, epoch)
            writer3.add_scalar('Precision/train', precision, epoch)
            writer3.add_scalar('Recall/train', recall, epoch)
            writer3.add_scalar('F1/train', f1, epoch)
        elif idx == 4:
            writer4.add_scalar('Loss/train', avg_loss_train, epoch )
            writer4.add_scalar('Accuracy/train', accuracy, epoch)
            writer4.add_scalar('Precision/train', precision, epoch)
            writer4.add_scalar('Recall/train', recall, epoch)
            writer4.add_scalar('F1/train', f1, epoch)
        elif idx == 5:
            writer5.add_scalar('Loss/train', avg_loss_train, epoch )
            writer5.add_scalar('Accuracy/train', accuracy, epoch)
            writer5.add_scalar('Precision/train', precision, epoch)
            writer5.add_scalar('Recall/train', recall, epoch)
            writer5.add_scalar('F1/train', f1, epoch)

        # 记录损失、准确率、精确率、召回率到 TensorBoard（统一添加标签用于区分不同优化器）
        precision, recall, accuracy, f1 = evaluate_model(model, test_loader)
        if idx ==1:
            writer1.add_scalar('Loss/test', avg_loss_test, epoch )
            writer1.add_scalar('Accuracy/test', accuracy, epoch)
            writer1.add_scalar('Precision/test', precision, epoch)
            writer1.add_scalar('Recall/test', recall, epoch)
            writer1.add_scalar('F1/test', f1, epoch)
        elif idx == 2:
            writer2.add_scalar('Loss/test', avg_loss_test, epoch )
            writer2.add_scalar('Accuracy/test', accuracy, epoch)
            writer2.add_scalar('Precision/test', precision, epoch)
            writer2.add_scalar('Recall/test', recall, epoch)
            writer2.add_scalar('F1/test', f1, epoch)
        elif idx == 3:
            writer3.add_scalar('Loss/test', avg_loss_test, epoch )
            writer3.add_scalar('Accuracy/test', accuracy, epoch)
            writer3.add_scalar('Precision/test', precision, epoch)
            writer3.add_scalar('Recall/test', recall, epoch)
            writer3.add_scalar('F1/test', f1, epoch)
        elif idx == 4:
            writer4.add_scalar('Loss/test', avg_loss_test, epoch )
            writer4.add_scalar('Accuracy/test', accuracy, epoch)
            writer4.add_scalar('Precision/test', precision, epoch)
            writer4.add_scalar('Recall/test', recall, epoch)
            writer4.add_scalar('F1/test', f1, epoch)
        elif idx == 5:
            writer5.add_scalar('Loss/test', avg_loss_test, epoch )
            writer5.add_scalar('Accuracy/test', accuracy, epoch)
            writer5.add_scalar('Precision/test', precision, epoch)
            writer5.add_scalar('Recall/test', recall, epoch)
            writer5.add_scalar('F1/test', f1, epoch)


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
    model = models[idx]  
    trained_model = train_model(optimizer, model, train_loader, test_loader, criterion,idx = idx+1, epochs=100 )


# 关闭 TensorBoard
writer1.close()
writer2.close()
writer3.close()
writer4.close()
writer5.close()

