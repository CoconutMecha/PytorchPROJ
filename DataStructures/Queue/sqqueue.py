# =================================
# File : sqqueue.py
# Description : 顺序队列
# Author : lbc
# CREATE TIME : 2023/4/24 10:54
# =================================

class SqQueue:
    def __init__(self):
        self.max_size = 10
        #队首 队尾指针
        self.front = 0
        self.rear = 0
        self.data = [None] * self.max_size

    #判断是否为空
    def empty(self):
        #当队首队尾指针重合时队伍为空
        return self.rear == self.front

    #入队
    def enqueue(self,e):
        #判断是否队满
        assert not self.rear == self.max_size
        #入队
        self.data[self.rear] = e
        self.rear += 1

    #出队
    def dequeue(self):
        #判断是否队空
        assert not self.empty()
        e = self.data[self.front]
        self.front += 1
        return e

    #获取队首元素
    def get_head(self):
        #判断是否队空,若为空返回None
        #assert not self.empty()
        if self.empty():
            return True
        else:
            return self.data[self.front]

if __name__ == '__main__':
    queue = SqQueue()
    queue.enqueue(1)
    queue.enqueue(2)
    queue.enqueue(3)
    print(queue.dequeue())
    print(queue.dequeue())
    print(queue.data)




