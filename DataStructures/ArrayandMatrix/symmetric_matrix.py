# =================================
# File : symmetric_matrix.py
# Description : 对称矩阵压缩存储
# Author : lbc
# CREATE TIME : 2023/4/24 16:19
# =================================

"""
实现主对角线对称矩阵的压缩存储
压缩存储方式: 使用一维数组仅存储下三角,设置索引,上三角元素根据下三角元素推荐
压缩后查找aij公式：
下三角: 1/2 * (i-1)[(1+i-1)] + j - 1 即目标前一行总元素个数加列数即下一行到目标元素的距离 因为数组下标从0开始 所以最后-1
上三角: 1/2 * (j-1)[(1+j-1)] + i - 1 交换i j

经压缩存储，可节省一半的存储空间
"""
class SymmetricMatrix:
    """
    初始化完成矩阵的压缩
    """
    def __init__(self,data):
        self.compress_matrix = []
        self.row = len(data)
        self.col = len(data[0])
        self.data = data
        #存储下三角,每行存储元素比前一行增加一个
        for row in range(self.row):
            for col in range(row + 1):
                self.compress_matrix.append(self.data[row][col])

    #压缩后下标的计算
    def index(self,row,col):
        assert 0 < row <= self.row and 0 < col <= self.col
        #如果待查找元素为下三角 如a11 a21 a31...
        if row >= col:
            k = row * (row - 1) * 1/2 + col - 1
        else:
            k = col * (col - 1) * 1/2 + row - 1
        assert 0 <= k <= len(self.compress_matrix)
        return self.compress_matrix[int(k)]

if __name__ == '__main__':
    sym = SymmetricMatrix(
        [
            [1,2,3],
            [2,1,8],
            [3,8,1]
        ]
    )

    print(sym.index(3,3))



