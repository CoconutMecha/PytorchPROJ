# =================================
# File : multi-classification.py
# Description : 多分类问题 使用MNIST数据集
# Author : lbc
# CREATE TIME : 2023/4/25 19:50
# ================================
import matplotlib.pyplot as plt
import torchvision
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torchvision.transforms import ToTensor
from matplotlib.pyplot import plot
from torch import nn

"""
加载手写数据集
使用TOTensor统一格式,完成归一化
"""
train_dataset = torchvision.datasets.MNIST("./data",transform=ToTensor(),download=True,train=True)
test_dataset = torchvision.datasets.MNIST("./data",transform=ToTensor(),download=True,train=False)

"""
查看前五张图片
"""
def img_show():
    img,label = next(iter(train_dl))
    plt.figure(figsize=(5,1))
    for i,img in enumerate(img[:5]):
        img_numpy = img.numpy()
        #展平 删除第一维
        img_numpy= np.squeeze(img_numpy)
        #配置子图
        plt.subplot(1,5,i+1)
        plt.imshow(img_numpy)
        #关闭坐标轴
        plt.axis("off")
    plt.show()
    print(label[:5])



"""
创建模型
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28*28,128)
        self.linear2 = nn.Linear(128, 88)
        self.linear3 = nn.Linear(88,64)
        self.linear4 = nn.Linear(64,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    def forward(self,input):
        x = input.view(-1,1*28*28)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        #未激活输出,供交叉熵损失使用
        logits = self.linear4(x)
        return logits

"""
将模型添加到可用设备
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model().to(device)

"""
训练相关参数
"""
epochs = 10
batch_size = 60
#batchs = len(train_dataset)//batch_size = len(dl.dataset)
lr = 0.001
#多分类问题使用交叉熵损失
loss_fn= nn.CrossEntropyLoss()
opt = torch.optim.SGD(model.parameters(),lr=lr)


"""
创建DataLoader
"""
train_dl = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dl = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)


"""
训练函数
"""
def train(train_dl,model,loss_fn,opt):
    #总数据量，可用于求平均值
    size = len(train_dl.dataset)
    #每轮训练所包含批次
    batchs = len(train_dl)

    correct,train_loss = 0,0
    for img, label in train_dl:
        img = img.to(device)
        label = label.to(device)
        label_pred = model(img)
        loss = loss_fn(label_pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            correct += (label_pred.argmax(1) == label).type(torch.int32).sum().item()
            train_loss += loss.item()
    #准确率
    correct /= size
    #平均损失
    train_loss /= batchs
    return correct , train_loss

"""
测试函数
"""
def test(test_dl,model,loss_fn):
    #总数据量，可用于求平均值
    size = len(test_dl.dataset)
    #每轮训练所包含批次
    batchs = len(test_dl)
    correct,test_loss = 0,0
    with torch.no_grad():
        for img, label in test_dl:
            img = img.to(device)
            label = label.to(device)
            label_pred = model(img)
            loss = loss_fn(label_pred, label)
            correct += (label_pred.argmax(1) == label).type(torch.int32).sum().item()
            test_loss += loss.item()

    #准确率
    correct /= size
    #平均损失
    test_loss /= batchs
    return correct,test_loss

def fit(train_dl,test_dl,epochs,model,loss_fn,opt):
    #训练损失
    train_loss = []
    #训练准确率
    train_acc = []
    #测试损失
    test_loss = []
    #测试准确率
    test_acc = []
    #开始训练
    for epoch in range(epochs):
        train_acc_epoch,train_loss_epoch = train(train_dl,model,loss_fn,opt)
        test_acc_epoch, test_loss_epoch = test(test_dl,model,loss_fn)
        train_loss.append(train_loss_epoch)
        train_acc.append(train_acc_epoch)
        test_loss.append(test_loss_epoch)
        test_acc.append(test_acc_epoch)

        #创建模板
        template = ("epoch: {:3d} ,train_loss:{:.5f} ,train_acc:{:.3f} , test_loss{:.5f} , test_acc:{:.3f}")
        print(template.format(epoch, train_loss_epoch, train_acc_epoch, test_loss_epoch, test_acc_epoch))
    print("train finish!")

    #返回损失和准确率用作绘图
    return train_loss,train_acc,test_loss,test_acc



train_loss,train_acc,test_loss,test_acc = fit(train_dl,test_dl,epochs,model,loss_fn,opt)
"""
绘图
"""
#配置子图
plt.figure(figsize=(20,10))
#绘制1-2中第一张
plt.subplot(1,2,1)
plt.plot(range(epochs),train_loss,label="train_loss")
plt.plot(range(epochs),test_loss,label="test_loss")
plt.legend()
#绘制1-2中第二张
plt.subplot(1,2,2)
plt.plot(range(epochs),train_acc,label="train_acc")
plt.plot(range(epochs),test_acc,label="test_acc")
plt.legend()
plt.show()
