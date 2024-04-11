import torch
from rustfrecord import sum_as_string, add_one

v = sum_as_string(4, 5)
print(v)

t = torch.Tensor([10,20,30])
v = add_one(t)
print (v)