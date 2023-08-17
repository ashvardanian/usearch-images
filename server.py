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


def _open_dataset(dir: os.PathLike) -> Dataset:
    print(f"Loading dataset: {dir}")
    vectors = load_matrix(
        os.path.join(dir, "images.uform-vl-multilingual-v2.fbin"),
        view=True,
    )
    count = vectors.shape[0]
    ndim = vectors.shape[1]
    print(f"Loaded {count}x {ndim}-dimensional vectors")
    index = Index(
        ndim=ndim,
        metric=MetricKind.Cos,
        path=os.path.join(dir, "images.uform-vl-multilingual-v2.usearch"),
    )

    if len(index) == 0:
        print("Will reconstruct the index!")
        index.add(None, vectors, log=True)
        index.save()

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


_model = get_model("unum-cloud/uform-vl-multilingual-v2")
_datasets = {
    name: _open_dataset(os.path.join("data", name)) for name in ("unsplash25k",)
}


def find_vector(dataset: str, vector: np.ndarray, count: int = 10) -> List[str]:
    vector = vector.flatten()
    assert dataset in _datasets.keys()
    dataset_object = _datasets[dataset]
    matches: Matches = dataset_object.index.search(vector, count)
    ids: np.ndarray = matches.keys.flatten()
    return [dataset_object.uris[id] for id in ids]


def sample_images(dataset_name: str, count: int = 10) -> List[str]:
    dataset = _datasets[dataset_name]
    images = list(np.random.choice(dataset.uris, count))
    return images


def find_with_vector(dataset: str, query: str, count: int) -> List[str]:
    """For the given `query` ASCII vector returns the URIs of the most similar images"""
    return find_vector(dataset, _ascii_to_vector(query), count)


def find_with_text(dataset: str, query: str, count: int) -> List[str]:
    """For the given `query` string returns the URIs of the most similar images"""
    if query is None or len(query) == 0:
        return sample_images(dataset, count)

    text_data = _model.preprocess_text(query)
    text_embedding = _model.encode_text(text_data).detach().numpy()
    return find_vector(dataset, text_embedding, count)


def find_with_image(dataset: str, query: Image, count: int) -> List[str]:
    """For the given `query` image returns the URIs of the most similar images"""
    image_data = _model.preprocess_image(query)
    image_embedding = _model.encode_image(image_data).detach().numpy()
    return find_vector(dataset, image_embedding, count)


def size(dataset: str) -> int:
    """Number of entries in the index"""
    return len(_datasets[dataset].index)


def dimensions(dataset: str) -> int:
    """Number of dimensions in vectors"""
    return _datasets[dataset].index.ndim


if __name__ == "__main__":
    server = Server()
    server(find_with_vector)
    server(find_with_text)
    server(find_with_image)
    server(size)
    server(dimensions)
    server.run()
