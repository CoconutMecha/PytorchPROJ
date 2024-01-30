# =================================
# File : linkqueue.py
# Description : 链式队列
# Author : lbc
# CREATE TIME : 2023/4/24 11:16
# =================================

class LinkNode:
    def __init__(self,data = None):
        #初始化队首队尾节点
        self.data = data
        self.next = None

class LinkQueue:
    def __init__(self):
        self.front = LinkNode()
        self.rear = self.front

    #判断是否为空
    def empty(self):
        return self.front == self.rear

    #入队
    def enqueue(self,e):
        #链表非连续存储，不不判断队满
        #向队尾加入元素
        node = LinkNode(e)
        self.rear.next = node
        self.rear = node

    #出队
    def dequeue(self):
        #判断是否为空
        assert not self.empty()
        pnode = self.front.next
        self.front.next = self.front.next.next
        #删除后被删除节点指针清空
        pnode.next = None
        #当队列删除至最后一个节点时需要把队尾指针复位
        if pnode == self.rear:
            self.rear = self.front

        return pnode.data

    #获取队首元素
    def get_head(self):
        # 判断是否为空
        assert not self.empty()
        return self.front.next.data

if __name__ == '__main__':
    linkq = LinkQueue()
    linkq.enqueue(1)
    linkq.enqueue(2)
    linkq.enqueue(3)
    print(linkq.get_head())
    print(linkq.dequeue())
    print(linkq.get_head())



