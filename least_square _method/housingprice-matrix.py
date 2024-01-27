import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
使用矩阵乘法拟合
"""
"""
pandas取值 列在前行在后
data[col][row]
"""
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

data = load_data()

#转换ndarray
X = np.array(data[0])
Y = np.transpose([np.array(data[1])])

# print(X.shape,Y.shape)
# print(Y)

"""
求解
"""
XT = np.transpose(X)
#np.linalg.solve求解第一个元素即自变量需要是方阵，不能使用result2 = np.linalg.solve(X,Y)直接求解
result1 = np.linalg.solve(XT.dot(X),XT.dot(Y))
#result2 = np.linalg.solve(X,Y)
#print(result1)


"""
输出
"""
#print(list(X[0].dot(result1)))
print(X[1].dot(result1))
print(Y[1])
print(list(X[0:100].dot(result1)))
print(list(Y[0:100]))
plt.plot(range(0,100),list(Y[0:100]))
plt.plot(range(0,100),list(X[0:100].dot(result1)))
plt.show()




