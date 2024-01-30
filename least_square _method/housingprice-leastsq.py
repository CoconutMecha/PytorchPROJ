import numpy as np
from scipy.optimize import leastsq
import pandas as pd
import matplotlib.pyplot as plt

"""
使用最小二乘法拟合
"""

#导入数据
def load_data():
    X=[]
    Y=[]
    data = pd.read_csv("housing.csv", header=None)
    # 选取3,14列
    #从第二行开始，删除标题
    for row in range(1,len(data)):
        temp = []
        for col in range(2,14):
            temp.append(float(data[col][row]))
        X.append(temp)

    for row in range(1, len(data)):
        Y.append(float(data[1][row]))


    return(X,Y)


#转换ndarray
data = load_data()
X = np.array(data[0])
Y = np.transpose([np.array(data[1])])
#print(X.shape[1])



#定义初始函数
def func(x,p):
    #p = np.random.randn(1,12)
    p = np.array(p)
    #取axis=1按行求和,需要保证y的值是一个一维数组
    y = np.sum(p*x,axis=1)
    return y

#定义误差函数
def residuals(p,y,x):
    return y - func(x,p)

print(X.shape,Y.shape)
#设置x样本
x = X

#设置带噪声的Y样本
#np.random.randn(X.shape)生成和X形状相同的噪声,X.shape返回元组不能直接写入
#将Y转换成一维ndarray
y = Y.reshape(4600,)+np.random.randn(4600,)


#设置初始值
#p0 = [3.0,1.5,1340,7912,1.5,0,0,3,1340,0,1955,2005]
p0 = np.random.randn(12)
#需要保证y的值是一个一维数组
result = leastsq(residuals,p0,args=(y,x))

print("权重为："+str(result[0]))
print(data[1][0:100])
print(func(x[0:100],list(result)[0]))
plt.plot(range(0,300),data[1][0:300])
plt.plot(range(0,300),func(x[0:300],list(result)[0]))
plt.legend(["real","pre"])
plt.show()

"""
备注：
无论计算过程如何变化，要保证输出值y是一维的
当有明确的x,y值时无需如样例中设置样本和真实函数
本例仍未得到理想效果
"""