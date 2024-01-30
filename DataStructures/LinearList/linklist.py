# =================================
# File : linklist.py
# Description : 单链表
# Author : lbc
# CREATE TIME : 2023/4/6 19:37
# ================================

"""
定义节点
"""
class LinkNode:
    def __init__(self,data = None):
        self.data = data
        self.next = None


"""
单链表节点类
"""

class LinkList:
    def __init__(self):
        self.head = LinkNode()
        self.head.next = None

    #根据list使用头插法创建链表
    def create_linklist_head(self,list):
        for item in list:
            node = LinkNode(item)
            node.next = self.head.next
            self.head.next = node

    #根据list使用尾插法创建链表
    def create_linklist_tail(self,list):
        #循环移动pnode指针找到最后一个节点位置
        pnode = self.head
        while pnode.next is not None:
            pnode = pnode.next

        #执行尾插法
        for item in list:
            node = LinkNode(item)
            pnode.next = node
            #指针永远在末尾
            pnode = node
        #尾节点置None
        node.next = None

    #添加元素到末尾
    def add(self,item):
        #从头节点开始，循环找到链表末尾
        pnode = self.head
        node = LinkNode(item)
        while pnode.next is not None:
            pnode = pnode.next
        pnode.next = node

    #获取链表长度
    def size(self):
        #计数器count从0开始
        count = 0
        #指针节点从头节点开始
        pnode = self.head
        while pnode.next is not None:
            count += 1
            pnode = pnode.next
        return count

    #获取index位置元素
    def get_item(self,index):
        assert index >= -1
        #指针从头节点开始
        pnode = self.head
        #初始下标，即头节点下标为-1
        pindex = -1
        #循环至pindex和index相等时退出，此时不能取等若取等还会执行一轮循环；同时判断指针节点后置节点是否为空，若为空表示链表到达尽头
        while pindex < index and pnode is not None:
            pnode = pnode.next
            pindex += 1
        if pnode is not None:
            return pnode
        else:
            return None

    #设置index位置的元素
    def set_item(self,index,item):
        assert index >= 0
        node = self.get_item(index)
        if node:
            node.data = item
            return True
        else:
            return False

    #在index位置插入新元素
    def insert(self,index,item):
        #找到插入位置的前置节点
        pnode = self.get_item(index - 1)
        #确定找到的pnode不为空
        if pnode:
            node = LinkNode(item)
            node.next = pnode.next
            pnode.next = node
            return True
        else:
            return False

    #删除index位置的元素
    def delete(self,index):
        assert index >= 0
        #找到index的前一个元素
        pnode = self.get_item(index-1)
        #确定找到的节点不为空
        if pnode and pnode.next:
            pnode.next = pnode.next.next
            return True
        else:
            return False

    #输出链表中所有元素
    def display(self):
        #因为输出所有元素不包括头节点，所以指针节点从头节点后置开始
        pnode = self.head.next
        while pnode is not None:
            print(pnode.data,end=',')
            pnode = pnode.next



#测试单链表
if __name__ == '__main__':
    link = LinkList()
    # # 测试头插法
    # link.create_linklist_head([1, 2, 3, 4, 5])
    # # 头插法顺序相反
    # link.display()
    #测试尾插法
    link.create_linklist_tail([1,2,3,4,5])
    # 测试添加
    #link.add(100)
    #测试大小
    print(link.size())
    #测试get
    print(link.get_item(5))
    #link.set_item(6,100)
    #link.insert(1,100)
    link.delete(5)
    link.display()



