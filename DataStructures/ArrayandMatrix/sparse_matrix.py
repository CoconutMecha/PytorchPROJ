# =================================
# File : sparse_matrix.py
# Description : 稀疏矩阵压缩存储
# Author : lbc
# CREATE TIME : 2023/4/25 14:08
# =================================

"""
稀疏矩阵分布无规律，无法推导相关公式，所以使用三元组对象压缩存储(也可使用十字链表存储)
"""

#创建三元组
class TripleTuple:
    def __init__(self,row,col,value):
        self.row = row
        self.col = col
        self.value = value


#稀疏矩阵压缩类
class SparseMatrix:
    def __init__(self,data):
        self.compress_matrix = []
        self.data = data
        self.row = len(data)
        self.col = len(data[0])

        #压缩操作
        for row in range(self.row):
            for col in range(self.col):
                if self.data[row][col] != 0:
                    #矩阵下标从1开始所以存储后需加一
                    trituple = TripleTuple(row+1,col+1,data[row][col])
                    self.compress_matrix.append(trituple)
    #压缩后下标的计算
    def index(self,row,col):
        assert 0<row<=self.row and 0<col<=self.col
        #循环遍历三元组与输入row,col比较
        for item in self.compress_matrix:
            if row == item.row and col == item.col:
                return item.value
        return 0

if __name__ == '__main__':
    sp = SparseMatrix([
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ])

    print(sp.index(5, 5))