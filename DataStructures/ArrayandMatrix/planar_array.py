# =================================
# File : planar_array.py
# Description : 二维数组 分别实现按行存储和按列存储
# Author : lbc
# CREATE TIME : 2023/4/24 15:17
# =================================

"""
初始元素为位置为LOC(a11),每个元素占k个空间的m行n列二维数组按行存储求aij公式为
LOC(a11) + (i - 1) * n * k + (j - 1) * k = LOC(a11) + [(i - 1) * n + (j - 1)] * k
初始元素为位置为LOC(a11),每个元素占k个空间的m行n列二维数组按列存储求aij公式为
LOC(a11) + (j - 1) * m * k + (i - 1) * k = LOC(a11) + [(j - 1) * n + (i - 1)] * k
"""
class PlanarArray:
    """
    初始化
    data:存放源数据的列表
    n:若为按行存储 n为列数  若为按列存储 n为行数 或者可当作每行有几个，每列有几个
    priority: row为按行存储(默认) col为按列存储 非正确输入输出input error
    """
    def __init__(self,data,n,priority="row"):
        self.data = data
        self.n = n
        self.priority = priority

    #压缩后下标的计算
    def index(self,row,col):
        assert 0 < row <= self.row and 0 < col <= self.col
        #若为按行存储
        if self.priority == "row":
            #显然LOC(a11) = 0  k = 1 原公式变形如下
            result = (row - 1) * self.n + col - 1
            #判断result是否越界
        elif self.priority == "col":
            result = (col - 1) * self.n + row - 1
        else:
            return "input error"
        assert 0 <= result < len(self.data)
        return self.data[result]
        pass

"""
1,2,3
4,5,6

1,4
2,5
3,6
"""
if __name__ == '__main__':
    arr = PlanarArray([1,2,3,4,5,6],3,priority='row')
    print(arr.index(1,3))