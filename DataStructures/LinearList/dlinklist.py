# =================================
# File : dlinklist.py
# Description : 双向链表
# Author : lbc
# CREATE TIME : 2023/4/9 8:10
# ================================

#定义链表节点
class DlinkNode:
    def __init__(self,data=None):
        self.data = data
        self.next = None
        self.prior = None


#定义双向链表
class DlinkList:
    def __init__(self):
        self.head = DlinkNode
        self.head.next = None
        self.head.prior = None

    #头插法创建
    def create_dlink_head(self,list):
        for item in list:
            node = DlinkNode(item)
            node.next = self.head.next
            node.prior = self.head
            self.head.next = node
            if node.next:
                node.next.prior = node

    #尾插法创建
    def create_dlink_tail(self,list):
        #找到最后一个节点
        pnode = self.head
        while pnode.next is not None:
            pnode = pnode.next

        for item in list:
            node = DlinkNode(item)
            #直接与前置节点建立往返
            pnode.next = node
            node.piror = pnode
            #p重新指向最后一个节点
            pnode = node

    #在最后插入节点
    def add(self,item):
        pnode = self.head
        while pnode.next is not None:
            pnode = pnode.next
        node = DlinkNode(item)
        pnode.next = node
        node.prior = pnode

        
    #获取双链表的长度
    def size(self):
        #指针指向头节点
        pnode = self.head
        count = 0
        while pnode.next is not None:
            count += 1
            pnode = pnode.next
        return count

    #获取指定位置的元素
    def get_item(self,index):
        #指针指向头节点
        pnode = self.head
        count = -1
        while pnode is not None and count < index:
            pnode = pnode.next
            count += 1

        if pnode is not None:
            return pnode
        else:
            return None

    #设置指定位置元素
    def set_item(self,index,item):
        #断言，确定范围
        assert index >= 0

        node = self.get_item(index)
        if node is not None:
            node.data = item
            return True
        else:
            return False

    #在index位置插入元素
    def insert(self,index,item):
        assert index>=0
        node = DlinkNode(item)
        pnode = self.get_item(index - 1)
        if pnode is not None:
            node.next = pnode.next
            node.prior = pnode
            pnode.next = node
            if node.next:
                node.next.piror = node
                return True
            else:
                return False
    #删除 index位置的元素
    def delete(self,index):
        assert index >= 0
        pnode = self.get_item(index)
        if pnode is not None:
            pnode.piror.next = pnode.next
            if pnode.next is not None:
                pnode.next.piror = pnode.piror
            pnode.next = pnode.piror = None
            return True
        else:
            return False

    #输出所有的节点

    def display(self):
        pnode = self.head.next
        while pnode is not None:
            print(pnode.data,end=',')
            pnode = pnode.next


if __name__ == '__main__':
    dl = DlinkList()
    #测试头插法
    # dl.create_dlink_head([1,2,3,4,5,6])
    # dl.display()
    #测试尾插法
    dl.create_dlink_tail([1,2,3,4,5,6])
    # dl.display()
    #dl.add(100)
    #dl.display()
    #print(dl.size())
    #print(dl.get_item(10).data)
    #dl.set_item(7,100)
    dl.insert(0,100)
    dl.delete(6)
    dl.display()






