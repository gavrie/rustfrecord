import torch
from rustfrecord import sum_as_string, add_one, tfrecord_open

v = sum_as_string(4, 5)
print(v)

t = torch.Tensor([10,20,30])
v = add_one(t)
print (v)

filename = "data/002scattered.training_examples.tfrecord.gz"
f = tfrecord_open(filename, compressed=True)
