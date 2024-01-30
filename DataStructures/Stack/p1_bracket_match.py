# =================================
# File : p1_bracket_match.py
# Description : 
# Author : lbc
# CREATE TIME : 2023/4/21 11:23
# =================================
import sqstack

def bracket_match(brackets):
    sqs = sqstack.SqStack()
    for item in brackets:
        if item=='(' or item=='['or item=='{':
            sqs.push(item)
        elif item == ')':
            if sqs.empty() or sqs.get_top() != '(':
                return False
            sqs.pop()

        elif item == ']':
            if sqs.empty() or sqs.get_top() != '[':
                return False
            sqs.pop()

        elif item == '}':
            if sqs.empty() or sqs.get_top() != '{':
                return False
            sqs.pop()

    return sqs.empty()

if __name__ == '__main__':
    brackets = ""
    print(bracket_match(brackets))