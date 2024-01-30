# =================================
# File : mydeque.py
# Description : 双端队列 使用python库实现
# Author : lbc
# CREATE TIME : 2023/4/24 14:29
# =================================

from collections import deque

"""
常见方法:
"""
#创建不定长的双端队列
deq = deque()
#创建固定长度的双端队列
deq = deque(10)
#根据列表创建双端队列
deq = deque([1,2,3])


#清空双端队列中的元素
deq.clear()
#在双端队列右端入队
deq.append(1)
#在双端队列左端入队
deq.appendleft(1)
#在双端队列右端出队一个元素
deq.pop()
#在双端队列左端出队一个元素
deq.popleft()
#获取双端队列长度
len(deq)
