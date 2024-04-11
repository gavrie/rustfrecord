import torch

from rustfrecord import Reader

filename = "data/002scattered.training_examples.tfrecord.gz"
r = Reader(filename, compressed=True)


for (i, features) in enumerate(r):
    print(i, features['label'])
