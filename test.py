import torch

a = torch.ones([8, 4, 5, 6])
print('a =',a.size())
b = torch.ones([8, 4, 5, 6])
print('b =',b.size())
c = a+b
print('c =',c.size())
