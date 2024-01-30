# =================================
# File : SqStack.py
# Description : 
# Author : lbc
# CREATE TIME : 2023/4/21 11:00
# =================================

class SqStack:
    def __init__(self):
        #初始化为空的list
        self.data = []

    #判断是否为空
    def empty(self):
        if len(self.data) == 0:
            return True
        else:
            return False

    #元素e进栈
    def push(self,e):
        self.data.append(e)

    #元素出栈
    def pop(self):
        assert not self.empty()
        self.data.pop()

    #取栈顶元素
    def get_top(self):
        assert not self.empty()
        return self.data[-1]


if __name__ == '__main__':
    sqs = SqStack()
    sqs.push(1)
    sqs.push(2)
    sqs.push(3)
    print(sqs.get_top())
    sqs.pop()
    sqs.pop()
