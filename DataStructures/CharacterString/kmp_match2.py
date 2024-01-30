# =================================
# File : kmp_match2.py
# Description : kmp匹配算法,个人写法
# Author : lbc
# CREATE TIME : 2023/4/26 16:35
# =================================

class KMP_Match2:
    def __init__(self,pattern):
        self.pattern = pattern
        #创建next数组,对应标准kmp算法
        self.next = [None] * len(pattern)
        #创建nextval数组,对于优化kmp算法
        self.nextval = [None] * len(pattern)

    def generate_next(self):
        #生成next数组不需要目标串参与,仅需对模式串处理
        #模式串下标
        i = 0
        #存放前缀
        prefix = []
        #存放后缀
        suffix = []
        #next数组前两位始终是-1,0
        self.next[0] = -1
        self.next[1] = 0
        #从第二位开始遍历,求出每个子串的前后缀,并比较
        for i in range(2,len(self.pattern)-1):
            #求前缀
            pre = ""
            su = ""
            for j in range(i):
                pre += self.pattern[j]
                prefix.append(pre)
            #求后缀
            for j in range(-2,0):
                su += self.pattern[j]
                suffix.append(su)

            result = []
            for i in list(set(prefix) & set(suffix)):
                result.append(len(i))

            self.next.append(max(result))

    #根据next生成nextval
    def generate_nextval(self):
        pass


    def match(self, target):
        # i: 遍历目标串 j: 遍历模式串
        i, j = 0, 0
        while i < len(target) and j < len(self.pattern):
            if j == -1 or target[i] == self.pattern[j]:
                i += 1
                j += 1
            else:
                j = self.next[j]
        if j == len(self.pattern):
            return i - len(self.pattern)
        return -1


if __name__ == '__main__':
    kmp = KMP_Match2("abc")
    kmp.generate_next()
    print(kmp.match("qqwweerrabc"))

