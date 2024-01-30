# =================================
# File : csqqueue.py
# Description : 循环顺序队列 循环队列可有效解决假溢出现象
# Author : lbc
# CREATE TIME : 2023/4/24 13:53
# =================================

class CSQQueue:
    def __init__(self):
        self.max_size = 10
        self.data = [None] * self.max_size
        self.front = 0
        self.rear = self.front

    #判断是否为空
    def empty(self):
        return self.front == self.rear

    #判断是否队满
    def full(self):
        #此处利用循环队列性质
        return (self.rear + 1) % self.max_size == self.front

    #入队
    def enqueue(self,e):
        assert not self.full()
        self.data[self.rear] = e
        #队尾延长
        self.rear = (self.rear + 1) % self.max_size

    #出队
    def dequeue(self):
        assert not self.empty()
        e = self.data[self.front]
        self.front = (self.front + 1) % self.max_size
        return e

    #获取队首元素
    def get_head(self):
        assert not self.empty()
        return self.data[self.front]

    #获取队列长度,使用循环队列性质
    def size(self):
        return (self.rear - self.front + self.max_size) % self.max_size


if __name__ == '__main__':
    csq = CSQQueue()
    csq.enqueue(1)
    csq.enqueue(2)
    csq.enqueue(3)
    csq.enqueue(4)
    csq.enqueue(5)
    print(csq.size())
    print(csq.get_head())
    print(csq.dequeue())
    print(csq.get_head())
    print(csq.dequeue())
    print(csq.get_head())
    print(csq.full())

