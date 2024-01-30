# =================================
# File : clinklist.py
# Description : 
# Author : lbc
# CREATE TIME : 2023/4/18 11:22
# =================================


class CLinkNode:
    def __init__(self,data=None):
        self.data = data
        self.next = None


class CLinkList:
    def __init__(self):
        self.head = CLinkNode()
        self.head.next = self.head

    #根据列表使用头插法创建循环
    def create_clinklist_head(self,list):
        for item in list:
            node = CLinkNode(item)
            node.next = self.head.next
            self.head.next = node

    # 根据列表使用尾插法创建循环
    def create_clinklist_tail(self,list):
        pnode = self.head
        #找到末位节点
        while pnode.next != self.head:
            pnode = pnode.next
        #创建
        for item in list:
            node = CLinkNode(item)
            pnode.next = node
            #pnode指向末位
            pnode = node
        #给末位添加循环
        pnode.next = self.head

    #在末位添加元素
    def add(self,item):
        node = CLinkNode(item)
        pnode = self.head
        while pnode.next != self.head:
            pnode = pnode.next

        pnode.next = node
        node.next = self.head

    #获取长度
    def size(self):
        count = 0
        pnode = self.head
        while pnode.next != self.head:
            count += 1
            pnode = pnode.next

        return count

    #获取index节点
    def get_item(self,index):
        assert index >= -1
        count = -1
        pnode = self.head
        while count < index:
            pnode = pnode.next
            count += 1
            if pnode == self.head:
                break
        #判断是否越界，越界则为空
        if pnode != self.head:
            return pnode
        else:
            return None

    #设置index节点
    def set_item(self,index,item):
        assert index >= 0
        #调用get_item
        node = self.get_item(index)
        if node:
            node.data = item
            return True
        else:
            return False

    #在中间插入节点
    def insert(self,index,item):
        assert index >= 0
        node = CLinkNode(item)
        pnode = self.get_item(index - 1)
        #此处注意，为0时get_item会跳过循环，又因为循环链表判断是否过界是与头节点比较所以会直接输出None
        if index == 0:
            node.next = self.head.next
            self.head.next = node
            return True
        else:
            if pnode:
                node.next = pnode.next
                pnode.next = node
                return True
            else:
                return False

    #删除节点
    def delete(self,index):
        assert index >= 0
        pnode = self.get_item(index - 1)
        #同insert单独处理为0的情况
        if index == 0:
            self.head.next = self.head.next.next
            return True
        else:
            if pnode:
                if pnode.next == self.head:
                    return False
                else:
                    pnode.next = pnode.next.next
                    return True
            else:
                return False

    #显示
    def display(self):
        pnode = self.head.next
        while pnode != self.head:
            print(pnode.data,end=',')
            pnode = pnode.next

if __name__ == '__main__':
    cl = CLinkList()
    #cl.create_clinklist_head([1,2,3,4,5])
    cl.create_clinklist_tail([1,2,3,4,5])
    cl.add(100)
    print(cl.size())
    print(cl.get_item(0).data)
    cl.set_item(0,100)
    cl.insert(0,100)
    cl.delete(2)
    cl.display()





