# =================================
# File : fashion_mnist.py
# Description : 使用FashionMNIST数据集完成多分类
# Author : lbc
# CREATE TIME : 2023/4/28 18:49
# ================================
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import nn


"""
导入数据
"""
train_dataset = torchvision.datasets.FashionMNIST('./FashionMNIST',transform=ToTensor(),download=True,train=True)
test_dataset = torchvision.datasets.FashionMNIST('./FashionMNIST',transform=ToTensor(),download=True,train=False)

"""
数据大小为[60000, 28, 28]
"""
#torch.Size([60000, 28, 28])
# print(train_data.data.shape)

"""
定义训练常用参数
"""
epochs = 10
batch_size = 60
lr = 0.001
loss_fn = torch.nn.CrossEntropyLoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'



"""
创建dataloader
"""
train_dl = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dl = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)


"""
绘制部分图片
"""
def show_img():
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
        self.linear2 = nn.Linear(128,64)
        self.linear3 = nn.Linear(64,32)
        self.linear4 = nn.Linear(32,10)
        self.relu = nn.ReLU()

    def forward(self,input):
        x = self.linear1(input.view(-1,28*28))
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        logits = self.linear4(x)
        return logits


model = Model()
opti = torch.optim.Adam(model.parameters(),lr=lr)
def train(model,train_dl,opti,loss_fn):
    #有多少dataset就有多大
    size = len(train_dl.dataset)
    #有多少dataloader就有多少batch
    batchs = len(train_dl)

    train_loss,train_correct = 0,0

    for img,label in train_dl:
        img = img.to(device)
        label = label.to(device)
        label_pred = model(img)
        loss = loss_fn(label_pred,label)
        opti.zero_grad()
        loss.backward()
        opti.step()

        with torch.no_grad():
            train_loss += loss
            train_correct += (label_pred.argmax(1) == label).type(torch.int32).sum().item()

    #准确率
    train_correct /= size
    #平均损失
    train_loss /= batchs
    return train_correct , train_loss


def test(model, test_dl, loss_fn):
    # 有多少dataset就有多大
    size = len(train_dl.dataset)
    # 有多少dataloader就有多少batch
    batchs = len(train_dl)

    test_loss, test_correct = 0,0
    with torch.no_grad():
        for img, label in train_dl:
            img = img.to(device)
            label = label.to(device)
            label_pred = model(img)
            loss = loss_fn(label_pred, label)
            test_loss += loss
            test_correct += (label_pred.argmax(1) == label).type(torch.int32).sum().item()

    # 准确率
    test_correct /= size
    # 平均损失
    test_loss /= batchs
    return test_correct, test_loss

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
        train_acc_epoch,train_loss_epoch = train(model,train_dl,opti,loss_fn)
        test_acc_epoch, test_loss_epoch = test(model, test_dl, loss_fn)
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



train_loss,train_acc,test_loss,test_acc = fit(train_dl,test_dl,epochs,model,loss_fn,opti)
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







