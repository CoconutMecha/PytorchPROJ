# =================================
# File : linear_regression.py
# Description : 1.线性回归联系 数据集来源kaggle:Linear Regression
# Author : lbc
# CREATE TIME : 2023/4/5 21:10
# ================================

import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

data = pd.read_csv("./lr_data/train.csv")

"""
查看数据集
"""
# print(data.head())
# print(data.info())

"""
绘图观察数据集
"""
# plt.scatter(data.x,data.y)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

"""
采用ndarray数值
"""
X = data.x[0:300].values
Y = data.y[0:300].values


"""
分别转置成列向量
"""
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)

"""
转换成float_tensor
"""
X = torch.from_numpy(X).type(torch.FloatTensor)
Y = torch.from_numpy(Y).type(torch.FloatTensor)

"""
创建模型
"""
class LRModel(nn.Module):
    def __init__(self):
        super(LRModel,self).__init__()
        self.linear = nn.Linear(in_features=1,out_features=1)

    def forward(self,inputs):
        logits = self.linear(inputs)
        return logits

"""
实例化模型
"""
model = LRModel()
#print(model)

"""
均方误差损失
"""
loss_fun = nn.MSELoss()

"""
优化函数:随机梯度下降 输入: 模型参数 学习速率
"""
optimizer = torch.optim.SGD(model.parameters(),lr=0.0001)

"""
训练
"""
for epoch in range(30):
    for x,y in zip(X,Y):
        y_pred = model(x)
        loss = loss_fun(y_pred,y)
        #优化器会累计梯度，每次优化前梯度清零
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #优化参数
        optimizer.step()


#查看优化后模型参数
print(list(model.parameters()))
# print(list(model.named_parameters()))
# print(model.linear.weight)

"""
绘图查看结果
"""
plt.scatter(data.x,data.y)
plt.xlabel("x")
plt.ylabel("y")

plt.plot(X,model(X).detach().numpy(),c='r')#截断梯度，只获取值
plt.show()
