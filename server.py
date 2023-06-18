import os
from typing import List
from dataclasses import dataclass

import numpy as np
from PIL.Image import Image
from tqdm import tqdm

from ucall.rich_posix import Server
from usearch.index import Index, MetricKind, Matches
from usearch.io import load_matrix
from usearch.server import _ascii_to_vector
from uform import get_model


@dataclass
class Dataset:
    index: Index
    uris: list
    vectors: np.ndarray


def open_dataset(dir: os.PathLike) -> Dataset:
    print(f"Loading dataset: {dir}")
    vectors = load_matrix(os.path.join(dir, "images.fbin"), view=True)
    count = vectors.shape[0]
    ndim = vectors.shape[1]
    print(f"Loaded {count}x {ndim}-dimensional vectors")
    index = Index(
        ndim=ndim,
        metric=MetricKind.Cos,
    )
    index.load(os.path.join(dir, "images.usearch"))

    if len(index) == 0:
        print("Will reconstruct the index!")
        batch_size = 1024
        labels = np.arange(count)
        for i in tqdm(range(0, count, batch_size)):
            index.add(labels[i : i + batch_size], vectors[i : i + batch_size])
        index.save(os.path.join(dir, "images.usearch"))

    print(f"Loaded index for {len(index)}x {index.ndim}-dimensional vectors")
    assert count == len(index), "Number of vectors doesn't match"
    assert ndim == index.ndim, "Number of dimensions doesn't match"
    uris = open(os.path.join(dir, "images.txt"), "r").read().splitlines()
    print(f"Loaded {len(uris)}x links")

    return Dataset(
        index=index,
        uris=uris,
        vectors=vectors,
    )


model = get_model("unum-cloud/uform-vl-multilingual")
datasets = {
    name: open_dataset(os.path.join("datasets", name))
    for name in ("unsplash25k", "cc3m")
}
server = Server()


def find_vector(dataset_name: str, vector: np.ndarray, count: int = 10) -> List[str]:
    vector = vector.flatten()
    assert dataset_name in datasets.keys()
    dataset = datasets[dataset_name]
    matches: Matches = dataset.index.search(vector, count)
    ids: np.ndarray = matches.labels.flatten()
    return [dataset.uris[id] for id in ids]


@server
def find_with_vector(dataset: str, query: str, count: int) -> List[str]:
    """For the given `query` ASCII vector returns the URIs of the most similar images"""
    return find_vector(dataset, _ascii_to_vector(query), count)


@server
def find_with_text(dataset: str, query: str, count: int) -> List[str]:
    """For the given `query` string returns the URIs of the most similar images"""
    text_data = model.preprocess_text(query)
    text_embedding = model.encode_text(text_data).detach().numpy()
    return find_vector(dataset, text_embedding, count)


@server
def find_with_image(dataset: str, query: Image, count: int) -> List[str]:
    """For the given `query` image returns the URIs of the most similar images"""
    image_data = model.preprocess_image(query)
    image_embedding = model.encode_image(image_data).detach().numpy()
    return find_vector(dataset, image_embedding, count)


@server
def size(dataset: str) -> int:
    """Number of entries in the index"""
    return len(datasets[dataset].index)


@server
def dimensions(dataset: str) -> int:
    """Number of dimensions in vectors"""
    return datasets[dataset].index.ndim


server.run()
