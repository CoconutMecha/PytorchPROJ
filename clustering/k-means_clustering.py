import numpy as np
import matplotlib.pyplot as plt

"""
k-means 二维数据
"""


"""
1.生成数据
"""
# #随机生成三类数据
# data1 = np.random.uniform(low=0, high=1, size=500).reshape([250,2])
# data2 = np.random.uniform(low=1, high=2, size=500).reshape([250,2])
# data3 = np.random.uniform(low=2, high=3, size=500).reshape([250,2])
# #拼接以上三个
# alldata = np.concatenate((data1,data2,data3),axis=0)
# #打乱
# np.random.shuffle(alldata)

alldata = np.random.rand(500,2)


#生成三个质心
centerpoint = np.random.rand(3,2)
# 三个空列表存放分类号的三类数据
classifylist = [[], [], []]
#定义距离阈值
threshold = 0.3


"""
2.计算相关函数
"""
#计算每个点与各个质心的距离
#1.采用欧式距离
def distance_o(p1,p2):
    #print(p2)
    return np.sqrt(
        (p1[0]-p2[0])**2 + ((p1[1]-p2[1])**2)
    )

#2.采用曼哈顿距离
def distance_m(p1,p2):
    return np.abs(p1[0]-p2[0]) + np.abs(p1[1]-p2[1])


#计算每个点和质心的距离若小于阈值则归为一类
def classify(centerpoint,data):
    # 定义三类数据
    temp_classifylist = [[], [], []]
    # 分类编号
    list_num = 0
    for center in centerpoint:
        # print(center)
        # print(list_num)
        for point in data:
            #采用欧式距离
            if(distance_o(point,center)) <=threshold:
                temp_classifylist[list_num].append(point)
        list_num += 1
    return temp_classifylist

#根据分类结果重新确定质心，使用平均值
def update_center(classifylist):
    centerpoint = [[],[],[]]
    # 定义分类编号
    list_num = 0
    average = [0, 0]
    for data in classifylist:
        for point in data:
            average[0] += point[0]
            average[1] += point[1]
        average[1] = average[1] / len(data)
        average[0] = average[0] / len(data)
        """
        注意 list名表示地址，不能直接赋值会互相影响
        若赋值，赋值后更新值重新赋予地址或者使用list.copy()函数构造一个副本
        """
        centerpoint[list_num] = average
        #平均值置空，计算下一个点
        #average = [0, 0]
        list_num += 1
    return centerpoint

"""
3.开始分类
"""
#第一次分类
classifylist = classify(centerpoint,alldata)

#重新计算并更新质心
centerpoint = update_center(classifylist)

#使用新质心再次分类，直到前后两次质心距离相差小于0.1
while(True):
    #分类
    classifylist = classify(centerpoint, alldata)
    # 计算前后两次质心距离
    center1_d = distance_o(update_center(classifylist)[0],centerpoint[0])
    center2_d = distance_o(update_center(classifylist)[1], centerpoint[1])
    center3_d = distance_o(update_center(classifylist)[2], centerpoint[2])
    if(center1_d<0.1 and center2_d<0.1 and center3_d<0.1):
        break
    # #重新计算并更新质心
    centerpoint = update_center(classifylist)


#取出每一类的横纵坐标，从左至右依次为坐标、类别,质心点放在末尾
#定义不同类别颜色，0,1,2代表分类 3代表质心点
color = ["blue","yellow","red","black"]
x = []
y = []
#定义迭代下标
listnum = 0

#处理所有点
for data in classifylist:
    for point in data:
        x.append([point[0],listnum])
        y.append([point[1],listnum])
    listnum+=1
    if(listnum>2): break

#处理质心
for data in centerpoint:
    x.append([data[0],3])
    y.append([data[1],3])


print("最终质心为：",centerpoint)
#将x,y转换成ndarray 其第一列为坐标，第二列为类别
x = np.array(x)
y = np.array(y)


#绘制结果

#生成两个子图
#1.所有点
plt.subplot(2,1,1)
#设置坐标轴单位 xmin xmax ymin ymax
#plt.axis([-1,1,-1,1])
plt.scatter(alldata[:,0],alldata[:,1],c="blue",s=20)

#2.处理后
#获取对应颜色-转换成ndarray使用ndarray特性获取第二列
color = np.array(color)

plt.subplot(2,1,2)
#plt.axis([-0.25,1,-0.5,1.5])
plt.scatter(x[:,0],y[:,0],c=color[x[:,1].astype(int)],s=20,alpha=0.7)
plt.show()
print(centerpoint)