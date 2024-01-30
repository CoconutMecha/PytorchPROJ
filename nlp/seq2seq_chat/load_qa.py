# =================================
# File : load_qa.py
# Description : 
# Author : lbc
# CREATE TIME : 2024/1/28 11:03
#生成question和answer
# ================================

allstr = open("./data/xiaohuangji50w_fenciA.conv",'r',encoding="utf-8")
answer = open("./data/answer.txt","w+",encoding='utf-8')
question = open("./data/question.txt","w+",encoding="utf-8")

strlist = allstr.readlines()
#去掉“/”
for i in range(100):
    strlist[i] = strlist[i].replace("/","")


stack1 = []
for i in range(20):
    stack1.append(strlist[i])
    if("E\n" in strlist[i]):
        if "E\n" in stack1[-1] and len(stack1)>1:

            question.write(stack1[-2][1:].replace("\n"," "))
            answer.write(stack1[-3][1:].replace("\n"," "))
            stack1 = ["E\n"]



