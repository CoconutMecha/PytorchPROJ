# =================================
# File : spaceship_test.py
# Description : kaggle competition : Spaceship Titanic (v2.0 test)
# Author : lbc
# CREATE TIME : 2023/4/30 15:33
# ================================
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt

url = "./data/test.csv"
data = pd.read_csv(url)

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



def load_test_data(data):
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

    return data


test_data = load_test_data(data)

test_x = torch.from_numpy(data.values).type(torch.float32)
print(test_x.shape)

model = torch.load("./model_save2.0")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
test_x = test_x.to(device)
result = model(test_x)
result = (result > 0.5)


#写入
w_url = "./submission/test3.csv"

result_np = result.cpu().numpy().reshape(1,-1)[0]
result_df = pd.DataFrame({"Transported":result_np})
result_df.to_csv(w_url,index=False)
