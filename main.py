import torch
from rustfrecord import Reader

filename = "data/002scattered.training_examples.tfrecord.gz"
r = Reader(filename, compressed=True)


# TODO: Implement Python __next__ method
i=0

while True:
    i+=1
    features = r.next()
    if features is None:
        break
    print(i, features['label'])
