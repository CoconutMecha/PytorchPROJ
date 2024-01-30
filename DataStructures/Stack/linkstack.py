# =================================
# File : linkstack.py
# Description : 
# Author : lbc
# CREATE TIME : 2023/4/21 11:08
# =================================

class LinkNode:
    def __init__(self,data=None):
        self.data = data
        self.next  =None


class LinkStack:
    def __init__(self):
        self.head = LinkNode()
        self.next = None

    #判断是否为空
    def empty(self):
        if self.head.next == None:
            return True
        else:
            return False

    #加入元素e
    def push(self,e):
        node = LinkNode(e)
        node.next = self.head.next
        self.head.next = node

    #pop
    def pop(self):
        #assert not self.empty()
        assert self.head.next is not None
        pnode = self.head.next
        self.head.next = not self.head.next
        return pnode.data

    #获取顶部元素
    def get_top(self):
        # assert not self.empty()
        assert self.head.next is not None
        return self.head.next.data