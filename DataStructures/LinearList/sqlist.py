# =================================
# File : sqlist.py
# Description : 数据结构线性表:顺序表
# Author : lbc
# CREATE TIME : 2023/4/3 18:49
# ================================

class Sqlist:
    def __init__(self):
        #初始容量
        self._initcapacity = 10
        #1.容量
        self._capacity = self._initcapacity
        #2.存放元素的容器
        self._data = [None] * self._capacity
        #3.当前大小
        self._size = 0



    #自动调整
    def auto_adjust_capacity(self):
        if self._capacity == self._size:
            # 重新设置容量，每次增加初始容量
            olddata = self._data
            self._capacity += self._initcapacity
            self._data = [None] * self._capacity
            for index, item in enumerate(olddata):
                self._data[index] = item
        elif self._size < self._capacity - self._initcapacity:
            olddata = self._data
            self._capacity -= self._initcapacity
            self._data = [None] * self._capacity
            for index in range(self._size):
                self._data[index] = olddata[index]

    #根据list创建sqlist
    def creat_list(self,a):
        for item in a:
            self.auto_adjust_capacity()
            self._data[self._size] = item
            self._size += 1

    #在末尾添加元素
    def add(self,item):
        self.auto_adjust_capacity()
        self._data[self._size] = item
        self._size += 1

    #获取当前顺序表长度
    def size(self):
        return self._size

    #在指定位置设置元素
    def set_item(self,index,item):
        assert 0 <= index <self._size
        self._data[index] = item

    #获取指定位置元素
    def get_item(self,index):
        assert 0 <= index <self._size
        return self._data[index]

    #在指定位置插入元素
    def insert(self,index,item):
        #[0,size]
        assert 0 <= index <= self._size
        self.auto_adjust_capacity()
        #插入操作
        for x in range(self._size,index,-1):
            self._data[x] = self._data[x-1]
        self._data[index] = item
        self._size += 1

    #删除指定位置元素
    def delete(self,index):
        #index [0,size)
        assert 0 <= index < self._size
        #后面元素向前移动依次覆盖
        for x in range(index,self._size,1):
            self._data[x] = self._data[x+1]
        self._size -= 1
        #自动调整容量
        self.auto_adjust_capacity()


    #输出顺序表中所有元素
    def display(self):
        for x in self._data:
            print(x, end=",")


#测试顺序表
if __name__ == '__main__':
    sq = Sqlist()
    sq.creat_list([1,2,3,4,5,6,7,8,9,10,11])
    sq.add(100)
    sq.delete(0)
    # sq.insert(0,20)
    # print(sq.get_item(2))
    # sq.set_item(2,300)
    sq.delete(1)
    sq.delete(2)
    print(sq.size())
    sq.display()
