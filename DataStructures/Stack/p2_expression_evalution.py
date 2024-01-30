# =================================
# File : p2_expression_evalution.py
# Description : 中缀转后缀
# Author : lbc
# CREATE TIME : 2023/4/23 14:55
# =================================
from sqstack import SqStack

"""
1.(入栈opts
2.) 运算符出栈并添加postexp,至(停止并丢弃
3.若栈顶运算符大于等于当前运算符，循环出栈，添加到postexp中，直到小于当前运算符，再将当前运算符入栈
4.后续数字转换成整形存放postexp  (此处注意数字处理完成需退一位，以防index越过预算符）
5.将栈中剩余操作符出栈并加入postexp列表中
"""
class Expression:
    def __init__(self,exp):
        self.exp = exp
        self.postexp = []

    def infix_to_postfix(self):
        opts = SqStack()
        #下标从0开始
        index = 0
        while index < len(self.exp):
            #取出每次循环遍历到的字符
            ch = self.exp[index]
            #1.判断是否为(
            if ch == "(":
                opts.push(ch)
            #2.判断是否为)
            elif ch == ")":
                while not opts.empty() and opts.get_top() != "(":
                    e = opts.pop()
                    self.postexp.append(e)
                #循环结束后,丢弃(
                opts.pop()
            #3.除括号外，+ -优先级最低，如遇到可一并处理
            #入当前元素为 + - 则除(外均满足栈顶元素大于等于当前元素，按循环出栈处理
            elif ch == "+" or ch == "-":
                while not opts.empty() and opts.get_top() != "(":
                    e = opts.pop()
                    self.postexp.append(e)
                opts.push(ch)
            #3.* / 是最大运算符，只需检查是否有相等的情况后可将其入栈
            elif ch == "*" or ch == "/":
                while not opts.empty():
                    e = opts.get_top()
                    if e == "*" or e == "/":
                        e = opts.pop()
                        self.postexp.append()
                    else:
                        break
                opts.push(ch)
            #4.处理数字
            else:
                num = ""
                #处于0-9之间即为数字，
                #不能直接加入postexp 需对两位数字进行处理
                while ch >= "0" and ch <= "9":
                    num += ch
                    #index自增，并据此重新获取ch
                    index += 1
                    #判断index是否越界
                    if index < len(self.exp):
                        #重新获取ch
                        ch = self.exp[index]
                    else:
                        #若越界则循环结束
                        break
                self.postexp.append(int(num))
                index -= 1
            index += 1
        #将栈中剩余操作符依次出栈并加入postexp列表中
        while not opts.empty():
            e = opts.pop()
            self.postexp.append(e)

    """
    根据后缀表达式求值：
    1.创建operand栈
    2.依次遍历后缀表达式中的元素
    3.若是操作数则入栈
    4.若是运算符则从栈中依次取出两个元素运算，先取出的元素放在后面,将运算后的结果入栈
    5.全部遍历完后栈中只剩一个元素,为运算结果
    """
    def post_fix_evaluate(self):
        operand = SqStack()
        for ch in self.postexp:
            #本例子只处理整数运算,所以操作数只有int
            if type(ch) == int:
                operand.push(ch)
            #反之为运算符，运算符只包含+-*/
            else:
                #首先执行出栈操作,先出栈的放在后面即op2,op1
                op2 = operand.pop()
                op1 = operand.pop()
                #根据运算符不同分别运算
                if ch == "+":
                    operand.push(op1 + op2)
                elif ch == "-":
                    operand.push(op1 - op2)
                elif ch == "*":
                    operand.push(op1 * op2)
                elif ch == "/":
                    operand.push(op1 / op2)
        #循环结束即全部遍历完,返回所剩元素即可
        return operand.pop()


if __name__ == '__main__':
    expression = Expression("(1+3)*2-(15-3)/3")
    expression.infix_to_postfix()
    print(expression.postexp)
    print(expression.post_fix_evaluate())



