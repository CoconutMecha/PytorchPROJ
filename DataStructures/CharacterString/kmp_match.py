# =================================
# File : kmp_match.py
# Description : kmp匹配算法
# Author : lbc
# CREATE TIME : 2023/4/26 10:22
# =================================

class KMP_Match:
    def __init__(self,pattern):
        self.pattern = pattern
        self.next = [-1] * len(pattern)
        self.nextval = [-1] * len(pattern)
    """
    创建next数组
    next数组表示模式串对应位置字符失配时,模式串指针在下次匹配时的位置
    """
    def generate_next(self):
        self.next[0] = -1
        #从0开始循环
        j = 0
        #默认-1表示不匹配
        k = -1
        while j < len(self.pattern) - 1:
            if k == -1 or self.pattern[j] == self.pattern[k]:
                j += 1
                k += 1
                self.next[j] = k
            else:
                k = self.next[k]
    """
    改进kmp 生成nextval数组
    """
    def generate_nextval(self):
        self.nextval[0] = -1
        j = 0
        k = -1
        while j < len(self.pattern) - 1:
            if k == -1 or self.pattern[j] == self.pattern[k]:
                j += 1
                k += 1
                if self.pattern[j] == self.pattern[k]:
                    self.nextval[j] = self.nextval[k]
                else:
                    self.nextval[j] = k
            else:
                k = self.nextval[k]

    def match(self,target):
        #i: 遍历目标串 j: 遍历模式串
        i,j = 0,0
        while i < len(target) and j < len(self.pattern):
            if j == -1 or target[i] == self.pattern[j]:
                i += 1
                j += 1
            else:
                j = self.nextval[j]
        if j == len(self.pattern):
            return i - len(self.pattern)
        return -1


if __name__ == '__main__':
    kmp = KMP_Match("abc")
    kmp.generate_nextval()
    print(kmp.match("qwerfgabc"))

