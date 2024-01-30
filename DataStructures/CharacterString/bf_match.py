# =================================
# File : bf_match.py
# Description : bf匹配算法
# Author : lbc
# CREATE TIME : 2023/4/25 15:02
# =================================

"""
传入两个参数: 目标串target 模式串pattern
匹配成功: 返回模式串在目标串的起始位置
匹配失败: 返回-1

时间复杂度:
假设模式串长度为m 目标串长度为n
最好的情况为一次就匹配成功，此时时间复杂度为O(m)
最坏的情况为最后一次才匹配成功，此时时间复杂度我O((n - m) * m) = O(nm)
"""

def bf_match(target,pattern):

    """
    #目标串起始指针
    targetp_start= 0
    #目标串结束指针
    target_end
    #模式串指针
    patternp = 0
    #最后一趟匹配，目标串起始指针位置
    targetp_start_final = 0
    """
    #目标串长度
    target_l = len(target)
    #模式串长度
    pattern_l = len(pattern)
    # 最后一趟匹配，目标串起始指针位置 +1是序列位置 -1是下标位置
    targetp_start_final = target_l - pattern_l +1 -1

    for targetp_start in range(targetp_start_final+1):
        # 目标串起始指针
        targetp_end = targetp_start
        # 模式串起始指针
        patternp = 0
        # 默认每一趟匹配结果为匹配
        matched = True
        while patternp < pattern_l:
            if target[targetp_end] == pattern[patternp]:
                targetp_end += 1
                patternp += 1
            else:
                #本趟不匹配
                matched = False
                break
        if matched:
            return targetp_start
    return -1


if __name__ == '__main__':
    print(bf_match("qweqweqwe", "qwe"))