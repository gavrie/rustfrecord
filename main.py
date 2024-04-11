import torch
from rustfrecord import tfrecord_open

filename = "data/002scattered.training_examples.tfrecord.gz"
t = tfrecord_open(filename, compressed=True)
print(t)
