import torch
from rustfrecord import Reader
from torch import Tensor


class TFRecordDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename: str, compressed: bool = True):
        super().__init__()
        self.filename = filename
        self.compressed = compressed

    def __iter__(self):
        # worker_info = torch.utils.data.get_worker_info()
        # if worker_info is None:  # single-process data loading, return the full iterator
        #     iter_start = self.start
        #     iter_end = self.end
        # else:  # in a worker process
        #     # split workload
        #     per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        #     worker_id = worker_info.id
        #     iter_start = self.start + worker_id * per_worker
        #     iter_end = min(iter_start + per_worker, self.end)
        reader = Reader(self.filename, compressed=self.compressed)
        return iter(reader)


def test_dataset():
    # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
    filename = "data/002scattered.training_examples.tfrecord.gz"
    ds = TFRecordDataset(filename, compressed=True)

    # Single-process loading
    print(list(torch.utils.data.DataLoader(ds, num_workers=0)))


def test_reader():
    filename = "data/002scattered.training_examples.tfrecord.gz"
    r = Reader(filename, compressed=True)

    for i, features in enumerate(r):
        """
        >>> print(i, features.keys())
        [
            "variant_type",
            "image/encoded",
            "image/shape",
            "variant/encoded",
            "label",
            "alt_allele_indices/encoded",
            "locus",
            "sequencing_type",
        ]
        """

        label: Tensor = features["label"]
        shape = torch.Size(tuple(features["image/shape"]))
        image: Tensor = features["image/encoded"][0].reshape(shape)

        print(i, label, image.shape)

        # break
