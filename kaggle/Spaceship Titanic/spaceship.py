# =================================
# File : spaceship.py
# Description : kaggle competition : Spaceship Titanic (v2.0 train)
# Author : lbc
# CREATE TIME : 2023/4/28 21:55
# ================================

"""
kaggle competition : Spaceship Titanic (v2.0 train)
思路：
1.决策树-通过计算信息增益及悲观后剪支的方式 待优化参数为阈值
2.多层感知机
经过比较多层感知机解决该数据集较具优势

特征工程注意点：
1.空值替换
2.字符串数据特征提取
3.编号等特殊数据特征提取
"""
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt

url = "./data/train.csv"
data = pd.read_csv(url)
#print(data.head())


"""
空值处理方法

判断该列是否存在空值：data[col].isnull().any()

对于空值可用均值替换 使用np.nanmean()忽略空值求均值

data[''].fillna('0.0', inplace = True) 替换空值

value_counts()   #统计这一列每个元素出现的个数
idxmax()  # 找出最多的元素
"""
def show_null_col():
    print("以下列有空行")
    for col in data.columns:
        print(col + " : ", data[col].isnull().any())
"""
PassengerId :  False
HomePlanet :  True
CryoSleep :  True
Cabin :  True
Destination :  True
Age :  True
VIP :  True
RoomService :  True
FoodCourt :  True
ShoppingMall :  True
Spa :  True
VRDeck :  True
Name :  True
Transported :  False
"""
"""
训练数据预处理
"""




def load_train_data(data):
    """
    #PassengerId列分为两部分Passenger 和 Id
    """
    data['Passenger'] = data['PassengerId'].map(lambda x: float(x[0:4]))
    data['Id'] = data['PassengerId'].map(lambda x: float(x[5:]))
    del data["PassengerId"]

    """
    #HomePlanet列使用one-hot处理
    """
    #空值处理将空值替换成该列出现次数最多的元素
    most_HomePlanet = data['HomePlanet'].value_counts().idxmax()
    data['HomePlanet'].fillna(most_HomePlanet, inplace = True)
    #one-hot处理并join
    HP = pd.get_dummies(data['HomePlanet'])
    data.join(HP)
    del data['HomePlanet']


    """
    # #CryoSleep列为bool值转华为0 1
    """
    #空值处理将空值替换为False
    data['CryoSleep'].fillna(False,inplace=True)
    data['CryoSleep'] = data['CryoSleep'].replace([True,False],[1,0])

    """
    #Cabin列按结构分成三列 C1 C2 C3 
    """
    #将空值替换为出现次数做多的
    most_Cabin = data['Cabin'].value_counts().idxmax()
    data['Cabin'].fillna(most_Cabin,inplace=True)
    #划分三列 字母替换ascii码表
    data['C1'] = data['Cabin'].map(lambda x: float(ord(str(x)[0])))
    data['C2'] = data['Cabin'].map(lambda x: float(str(x)[2]))
    data['C3'] = data['Cabin'].map(lambda x: float(ord(str(x)[4])))
    #删除原列
    del data['Cabin']

    """
    Destination列使用one-hot处理
    """
    #空值使用出现次数最多的替换
    most_Destination = data['Destination'].value_counts().idxmax()
    data['Destination'].fillna(most_Destination, inplace = True)
    #one-hot处理并join
    DS = pd.get_dummies(data['Destination'])
    data.join(DS)
    del data['Destination']

    """
    Age列保持原状，空值采用均值替换
    """
    #计算非空均值
    age_nanmean = np.nanmean(data['Age'].values)
    data['Age'].fillna(age_nanmean,inplace=True)

    """
    VIP列 保持原状，空值替换为False
    """
    data['VIP'].fillna(False,inplace=True)
    data['VIP'] = data['VIP'].replace([True,False],[1,0])

    """
    RoomService 列空值替换为均值
    """
    room_nanmean = np.nanmean(data['RoomService'].values)
    data['RoomService'].fillna(room_nanmean,inplace=True)

    """
    FoodCourt 列空值替换为均值
    """
    food_nanmean = np.nanmean(data['FoodCourt'].values)
    data['FoodCourt'].fillna(food_nanmean,inplace=True)

    """
    ShoppingMall 列空值替换为均值
    """
    shop_nanmean = np.nanmean(data['ShoppingMall'].values)
    data['ShoppingMall'].fillna(shop_nanmean,inplace=True)

    """
    Spa 列空值替换为均值
    """
    spa_nanmean = np.nanmean(data['Spa'].values)
    data['Spa'].fillna(spa_nanmean,inplace=True)

    """
    VRDeck 列空值替换为均值
    """
    vr_nanmean = np.nanmean(data['VRDeck'].values)
    data['VRDeck'].fillna(vr_nanmean,inplace=True)

    """
    Name 列 替换为姓出现的频率
    """
    #将空值替换为None
    data['Name'].fillna("None",inplace=True)
    #使用map方法去除该列的名字保留姓 若为空替换为0
    data['Name'] = data['Name'].map(lambda x: str(x).split()[1] if x != "None" else 0)
    #频率列表
    counts = data['Name'].value_counts()
    print(type(counts))
    #使用map方法去将值转化为频率 0替换成均值
    data['Name'] = data['Name'].map(lambda x: counts[str(x)] if x != 0 else counts.mean())


    """
    Transported列转化为 0，1
    """
    data['Transported'] = data['Transported'].replace([True,False],[1,0])
    return data



"""
创建模型
"""
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(14,128)
        self.linear2 = nn.Linear(128,100)
        self.linear3 = nn.Linear(100,128)
        self.linear4 = nn.Linear(128,64)
        self.linear5 = nn.Linear(64,32)
        self.linear6 = nn.Linear(32, 16)
        self.linear7 = nn.Linear(16, 8)
        self.linear8 = nn.Linear(8, 1)
        self.relu = nn.LeakyReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        x = self.linear1(input)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.leakyrelu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.leakyrelu(x)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)
        x = self.relu(x)
        x = self.linear8(x)
        x = self.sigmoid(x)

        return x

model = Model()

"""
训练参数
"""
epochs = 200
batch_size = 1024
lr = 0.001
opti = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=0.001)
loss_fn = torch.nn.BCELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

"""
创建dataset dataloader
"""
train_data = load_train_data(data)
#所有行都要，列取到最后一列之前
train_x = torch.from_numpy(data[[c for c in data.columns if c != "Transported"]].values).type(torch.float32)
#所有行都要，列取最后一列
train_y = torch.from_numpy(train_data["Transported"].values).reshape(-1,1).type(torch.float32)
print(train_x.shape)
print(train_y.shape)
print(data.shape)
train_dataset = TensorDataset(train_x,train_y)
train_dl = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

"""
平均准确率函数
"""
def accuracy(y_pred,y_true):
    y_pred = (y_pred>0.5).type(torch.int32)
    acc = (y_pred==y_true).type(torch.float32).mean()
    return acc

"""
训练
"""
train_x = train_x.to(device)
train_y = train_y.to(device)

# epoch_loss_result = []
# epoch_acc_result = []
def fit(model,epochs,train_dl,loss_fn,opti):
    model = model.to(device)
    for epoch in range(epochs):
        for x,y in train_dl:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred,y)
            opti.zero_grad()
            loss.backward()
            opti.step()

        with torch.no_grad():
            epoch_accuracy = accuracy(model(train_x), train_y)
            epoch_loss = loss_fn(model(train_x), train_y).data


            print("epoch:", epoch,
                  "loss:", epoch_loss, "accuracy:", epoch_accuracy,)

fit(model,epochs,train_dl,loss_fn,opti)

# for i in train_x:
#     print(model(i))

"""
保存模型
"""
torch.save(model,"./model_save2.0")