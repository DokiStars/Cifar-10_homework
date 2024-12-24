# Cifar-10_homework
使用ResNet系列网络测试不同优化器在Cifar-10数据集上的表现结果并分析

## 1.环境配置
···
python 3.x
torch  2.2.2
torchvision 0.17.2
sklearn 1.5.2
···

## 2.各python文件介绍
```
--1.1 model.py            包含resnet18/34/50/101/152共5个网络模型的结构，可通过如resnet18=ResNet18()获得网络结构
--1.2 train_resnet18.py   用于测试resnet18并使用5种不同的优化器在Cifar10数据集上进行训练，通过TensorBoard记录训练及测试过程中准确率及loss等的变化曲线
--1.3 train_resnets.py    使用Adam优化器，测试5种不同的resnet网络在Cifar10数据集上训练的过程，通过TensorBoard记录
--1.4 train_lr.py         使用resnet18网络，在SGD及Adam优化器上使用不同初始学习率进行训练，通过TensorBoard记录训练过程
--1.5 loss_b.py           由于前面实验存在测试集loss在一定epoch后异常增加的现象，使用flood方法以不同的b参数值来改进resnet18的训练过程，通过TensorBoard记录训练过程
```
##  3.经过以上文件运行后目录结构如下:
```
-----checkpoints 用于保存训练好的resnet网络权重
     data        保存下载好的Cifar10数据集
     runs        保存TensorBoard记录的训练过程数据文件
     loss_b.py 
     model.py 
     train_lr.py
     train_resnet18.py 
     train_resnets.py  
```
