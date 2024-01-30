# =================================
# File : Multi-layer-perceptron-2.py
# Description : 多层感知机预测HR数据完整代码 数据集来源 kaggle:HR Analytics
# Author : lbc
# CREATE TIME : 2023/4/23 21:06
# ================================

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

"""
读取数据集
"""
url = "./HR-Employee-Attrition.csv"
data = pd.read_csv(url)

"""
数据预处理
"""
#使用one-hot处理数据集中文字部分
BusinessTravel_0 = pd.get_dummies(data.BusinessTravel)
Department_0 = pd.get_dummies(data.Department)
EducationField_0 = pd.get_dummies(data.EducationField)
Gender_0 = pd.get_dummies(data.Gender)
JobRole_0 = pd.get_dummies(data.JobRole)
MaritalStatus_0 = pd.get_dummies(data.MaritalStatus)
Over18_0 = pd.get_dummies(data.Over18)
OverTime_0 = pd.get_dummies(data.OverTime)

#删除多列
data = data.drop(['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus','Over18','OverTime'],axis=1)

#使用处理后的列拼接替换
list = [BusinessTravel_0,Department_0,EducationField_0,Gender_0,JobRole_0,MaritalStatus_0,Over18_0,OverTime_0]
for item in list:
    data = data.join(item,rsuffix='_r')

#Attrition列 Yes-1 No-0
data['Attrition'] = data['Attrition'].replace(["Yes","No"],[1,0])


"""
划分数据集
"""
#划分全部输入输出数据 X Y
Y_data =  data.iloc[:,1:2].values
X_data =  data[[c for c in data.columns if c != 'Attrition']].values
Y = torch.from_numpy(Y_data).type(torch.float32)
X = torch.from_numpy(X_data).type(torch.float32)


#sklearn 划分训练集和测试集
train_x_data,test_x_data,train_y_data,test_y_data = train_test_split(X_data,Y_data)

train_x = torch.from_numpy(train_x_data).type(torch.float32)
test_x = torch.from_numpy(test_x_data).type(torch.float32)
train_y = torch.from_numpy(train_y_data).type(torch.float32)
test_y = torch.from_numpy(test_y_data).type(torch.float32)



"""
创建模型
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(57,128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self,input):
        x = self.linear1(input)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x


"""
设置训练相关参数
"""
model = Model()
epochs = 200
batch_size = 64
batches = train_x//batch_size
loss_fun = torch.nn.BCELoss()
lr = 0.000001
opt = torch.optim.Adam(model.parameters(),lr=lr)

#平均准确率函数
def accuracy(y_pred,y_true):
    y_pred = (y_pred > 0.6).type(torch.int32)
    acc = (y_pred == y_true).type(torch.float32).mean()
    return acc


"""
配置DataSet和DataLoader
"""
#创建DataSet,DataLoader
dataset = TensorDataset(train_x,train_y)
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
print(train_x.shape)


"""
训练
"""
for epoch in range(epochs):
    for x,y in dataloader:
        y_pred = model(x)
        loss = loss_fun(y_pred,y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        #训练平均准确率和损失
        epoch_accuracy = accuracy(model(train_x),train_y)
        epoch_loss = loss_fun(model(train_x),train_y).data
        #测试平均准确率和损失
        epoch_test_accuracy = accuracy(model(test_x), test_y)
        epoch_test_loss = loss_fun(model(test_x), test_y).data
        print("epoch:",epoch,
              "loss:",epoch_loss,"accuracy:",epoch_accuracy,
              "epoch_test_loss:",epoch_test_loss,"epoch_test_accuracy:",epoch_test_accuracy)

