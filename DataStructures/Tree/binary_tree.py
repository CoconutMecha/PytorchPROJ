# =================================
# File : binary_tree.py
# Description : 链式二叉树
# Author : lbc
# CREATE TIME : 2023/4/27 16:05
# =================================

"""
辅助顺序栈 用于辅助实现非递归的遍历
"""
class SqStack:
    def __init__(self):
        self.data = []

    def empty(self):
        if len(self.data) == 0:
            return True
        else:
            return False

    def push(self,e):
        self.data.append(e)

    def pop(self):
        assert not self.empty()
        return self.data.pop()

    def get_head(self):
        assert not self.empty()
        return self.data[-1]

"""
二叉树有顺序存储和链式存储两种方式
顺序存储存在一定局限性，如非完全二叉树会造成空间浪费，所二叉树的存储方式通常采用链式存储
"""
class BTNode:
    def __init__(self,data = None):
        self.data = data
        self.lchild = None
        self.rchild = None
        self.parent = None

    #访问某个节点
    def visit(self):
        print(self.data,end=',')

    """
    采用递归方法实现先序遍历
    递归方法可以理解为分每次调用都向更深一层推进，先输出左后输出右
    即
    1.先访问根节点
    2.再访问左子树
    3.最后访问右子树
    """
    def pre_order_recrusion(self):
        self.visit()
        if self.lchild:
            self.lchild.pre_order_recrusion()
        if self.rchild:
            self.rchild.pre_order_recrusion()

    """
    采用非递归方式实现
    1.根节点入栈
    栈不为空则循环：
    2.栈顶元素出栈
    2.若出栈元素有右节点则将其入栈。(因为栈是先进后出所以先右后左)
    3.若出栈元素有左节点则将其入栈
    以上可以理解为左节点始终在右节点上，即左节点出栈那么左节点的子节点就入栈，
    即左子树未处理完成栈便不会运行到右节点,而当栈为空是表示左右节点全部运行完也就完成了先序遍历
    """
    def pre_order_no_recrusion(self):
        stack = SqStack()
        #根节点入栈
        stack.push(self)
        #只要栈不为空就一直循环
        while not stack.empty():
            #出栈
            node = stack.pop()
            #读取
            node.visit()
            if node.rchild is not None:
                stack.push(self.data)
            if node.lchild is not None:
                stack.push(self.data)

    """
    采用递归的方法实现中序遍历
    1.先访问左子树
    2.再访问根节点
    3.最后访问右子树
    """
    def in_order_recrusion(self):
        #访问左子树
        if self.lchild is not None:
            self.lchild.in_order_recrusion()
        #访问根节点
        self.visit()
        #访问右子树
        if self.rchild is not None:
            self.rchild.in_order_recrusion()


    """
    非递归的中序遍历
    1.将根节点入栈
    循环：
    2.出栈，判断出栈节点是否有左节点，若有则入栈，若无则将当前元素出栈
    """
    def in_order_no_recrusion(self):
        stack = SqStack()
        node = self
        while not stack.empty() or node is not None:
            while node is not None:
                stack.push(node)
                node = node.lchild

            if not stack.empty():
                node = stack.pop()
                node.visit()
                #将右节点作为根节点
                node = node.rchild

    """
    后续遍历 采用递归方式
    后续遍历与前序遍历正好相反
    """
    def post_order_recrusion(self):
        if self.lchild:
            self.lchild.post_order_recrusion()
        if self.rchild:
            self.rchild.post_order_recrusion()

        self.visit()









if __name__ == '__main__':
    a = BTNode("A")
    b = BTNode("B")
    c = BTNode("C")
    d = BTNode("D")
    e = BTNode("E")

    a.lchild = b
    a.rchild = c
    b.parent = c.parent = a
    c.lchild = d
    c.rchild = e
    d.parent = e.parent = c

    #递归先序遍历
    #a.pre_order_recrusion()
    #非递归先序遍历
    #a.pre_order_recrusion()
    #递归中序遍历
    #a.in_order_recrusion()
    #非递归中序遍历
    #a.in_order_no_recrusion()

    #后续遍历递归
    a.post_order_recrusion()




