# =================================
# File : test.py
# Description : 
# Author : lbc
# CREATE TIME : 2023/4/10 21:52
# ================================

import torch
t = torch.Tensor([1,2])
#t = torch.ones(2,2)
t.requires_grad=True
x = t+2
y = x*2
z = y.mean()
z.backward()
print(t.grad)