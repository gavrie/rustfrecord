import torch
from rustfrecord import Reader
from torch import Tensor

# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)


class TFRecordDataset(torch.utils.data.IterableDataset):
    def __init__(self, filename: str, compressed: bool = True, features: list = None):
        super().__init__()
        self.filename = filename
        self.compressed = compressed
        self.features = features

    def __iter__(self):
        reader = Reader(
            self.filename,
            compressed=self.compressed,
            features=self.features,
        )
        return iter(reader)


def test_loader():
    filename = "data/002scattered.training_examples.tfrecord"
    ds = TFRecordDataset(
        filename,
        compressed=filename.endswith(".gz"),
        features=[
            "label",
            "image/encoded",
            "image/shape",
        ],
    )

    batch_size = 256
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    print()

    for i, batch in enumerate(loader):
        labels: Tensor = batch["label"]
        shapes: Tensor = batch["image/shape"]
        images: Tensor = batch["image/encoded"]

        j = 0
        label = labels[j]
        shape = torch.Size(tuple(shapes[j]))
        image = images[j].reshape(shape)
        print(f"batch={i} ({len(labels)}), {j=}, {label=}, {image.shape}")


def test_dataset():
    filename = "data/002scattered.training_examples.tfrecord"

    for _ in range(10):
        ds = TFRecordDataset(
            filename,
            compressed=filename.endswith(".gz"),
            features=[
                "label",
                "image/encoded",
                "image/shape",
            ],
        )

        print()

        for i, features in enumerate(ds):
            label: Tensor = features["label"]
            shape = torch.Size(tuple(features["image/shape"]))
            image: Tensor = features["image/encoded"][0].reshape(shape)

            if i % 1000 == 0:
                print(i, label, image.shape)


def test_reader():
    filename = "data/002scattered.training_examples.tfrecord"

    r = Reader(
        filename,
        compressed=filename.endswith(".gz"),
        features=[
            "label",
            "image/encoded",
            "image/shape",
        ],
    )

    print()

    for i, features in enumerate(r):
        label: Tensor = features["label"]
        shape = torch.Size(tuple(features["image/shape"]))
        image: Tensor = features["image/encoded"][0].reshape(shape)

        if i % 1000 == 0:
            print(i, label, image.shape)
