# =================================
# File : triangular_matrix.py
# Description : 三角矩阵压缩存储
# Author : lbc
# CREATE TIME : 2023/4/25 10:40
# =================================
"""
压缩存储主对角线下三角矩阵 m行n列
压缩方法:将下三角所有元素存储到数组中,将上三角元素存储到数组末尾,最后根据原索引重映射索引
重映射方法:
若待查找元素为aij则
当i <= j时为上三角元素 此时LOC(aij) = n * (n+1) * 1/2
当i > j时为下三角元素 此时LOC(aij) = i * (i-1) *1/2 + j - 1
"""

class TriMatrix:
    #压缩方法
    def __init__(self,data):
        #初始化原矩阵参数
        self.compress_matrix = []
        self.data = data
        self.row = len(data)
        self.col = len(data[0])

        #压缩下三角
        for row in range(self.row):
            for col in range(row + 1):
                self.compress_matrix.append(self.data[row][col])
        #压缩上三角
        self.compress_matrix.append(data[0][-1])

    #压缩后下标的计算
    def index(self,row,col):
        assert 0 < row <= self.row and 0 < col <= self.col
        #当待查找元素为下三角元素时
        if row >= col:
            k = row * (row - 1) * 1/2 + col - 1
        else:
            #k = -1 使用python特性也可解决,不用计算了
            k = (self.col * self.col) * 1/2

        #判断k是否越界
        assert  0 <= k < len(self.compress_matrix)
        return self.compress_matrix[int(k)]

if __name__ == '__main__':
    trim = TriMatrix([
        [1,2,2],
        [4,5,2],
        [7,8,9]
    ])

    print(trim.index(1,1))

