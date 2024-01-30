# =================================
# File : diagonal_matrix.py
# Description : 对角矩阵压缩存储
# Author : lbc
# CREATE TIME : 2023/4/25 13:38
# =================================
"""
主对角线对角矩阵压缩存储
压缩方法，仅存储非零元素 即|i-j| <= 1的元素
索引重映射为 k = 2 + 3 * ((i - 1) - 1) + (j - i) + 2 - 1 = 2 * i + j -3
"""
class DiagMatrix:
    def __init__(self,data):
        self.data = data
        self.compress_matrix = []
        self.row = len(data)
        self.col = len(data[0])

        #压缩操作
        for row in range(self.row):
            #对角矩阵第一行和最后一行均为两个元素，需单独出来
            if row == 0:
                #第一行存储前两个元素
                self.compress_matrix.append(self.data[0][0])
                self.compress_matrix.append(self.data[0][1])
            elif row == len(self.data) - 1:
                #最后一行存储后两个元素
                self.compress_matrix.append(self.data[self.row-1][self.col-1])
                self.compress_matrix.append(self.data[self.row-1][self.col-2])
            else:
                #其他行存储主对角线及其前后共三个元素
                for col in range(row-1,row+2):
                    self.compress_matrix.append(data[row][col])

    # 压缩后下标的计算
    def index(self,row,col):
        assert 0 < row <= self.row and 0 < col <= self.col
        #判断是否是对角元素,若是代入公式求新索引，若不是返回0
        if abs(row - col) <= 1:
            k = 2 * row + col - 3
            assert 0 <= k <= len(self.compress_matrix)
            return self.compress_matrix[int(k)]
        else:
            return 0
if __name__ == '__main__':
    diam = DiagMatrix([
        [1, 2, 0, 0, 0],
        [1, 2, 3, 0, 0],
        [0, 2, 3, 4, 0],
        [0, 0, 3, 4, 5],
        [0, 0, 0, 4, 5],
    ])

    print(diam.index(2,2))
