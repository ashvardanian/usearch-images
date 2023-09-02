import os
import re
import io
import base64
from typing import List
from dataclasses import dataclass

import numpy as np
from PIL import Image

from stringzilla import File, Strs
from ucall.rich_posix import Server
from usearch.index import Index, MetricKind, Matches
from usearch.io import load_matrix
from usearch.server import _ascii_to_vector
from uform import get_model


@dataclass
class Dataset:
    index: Index
    uris: Strs
    vectors: np.ndarray


def _open_dataset(dir: os.PathLike) -> Dataset:
    print(f"Loading dataset: {dir}")
    vectors = load_matrix(
        os.path.join(dir, "images.uform-vl-multilingual-v2.fbin"),
        view=True,
    )
    count = vectors.shape[0]
    ndim = vectors.shape[1]
    print(f"- loaded {count:,} x {ndim}-dimensional vectors")
    index = Index(ndim=ndim, metric=MetricKind.Cos)

    index_path = os.path.join(dir, "images.uform-vl-multilingual-v2.usearch")
    if os.path.exists(index_path):
        index.load(index_path)

    if len(index) == 0:
        print("Will reconstruct the index!")
        index.add(None, vectors, log=True)
        index.save(index_path)

    print(f"- loaded index for {len(index):,} x {index.ndim}-dimensional vectors")
    assert count == len(index), "Number of vectors doesn't match"
    assert ndim == index.ndim, "Number of dimensions doesn't match"
    uris: Strs = File(os.path.join(dir, "images.txt")).splitlines()
    print(f"- loaded {len(uris):,} links")

    return Dataset(
        index=index,
        uris=uris,
        vectors=vectors,
    )


_model = get_model("unum-cloud/uform-vl-multilingual-v2")
_datasets = {
    name: _open_dataset(os.path.join("data", name))
    for name in (
        "unsplash-25k",
        "cc-3m",
    )
}


def find_vector(dataset: str, vector: np.ndarray, count: int = 10) -> List[str]:
    vector = vector.flatten()
    assert dataset in _datasets.keys()
    dataset_object = _datasets[dataset]
    matches: Matches = dataset_object.index.search(vector, count)
    ids: np.ndarray = matches.keys.flatten()
    uris: List[str] = [str(dataset_object.uris[id]) for id in ids]
    return uris


def sample_images(dataset_name: str, count: int = 10) -> List[str]:
    dataset = _datasets[dataset_name]
    images = list(np.random.choice(dataset.uris, count))
    return images


def find_with_vector(dataset: str, query: str, count: int) -> List[str]:
    """For the given `query` ASCII vector returns the URIs of the most similar images"""
    return find_vector(dataset, _ascii_to_vector(query), count)


def find_with_text(
    dataset: str,
    query: str,
    count: int,
    rerank: bool = False,
) -> List[str]:
    """For the given `query` string returns the URIs of the most similar images"""
    if query is None or len(query) == 0:
        return sample_images(dataset, count)

    text_data = _model.preprocess_text(query)
    text_embedding = _model.encode_text(text_data).detach().numpy()
    uris = find_vector(dataset, text_embedding, count)

    if rerank:
        reranked = []
        for uri in uris:
            data: str = re.sub("^data:image/.+;base64,", "", uri)
            image = Image.open(io.BytesIO(base64.b64decode(data)))
            image_data = _model.preprocess_image(image)
            joint_embeddings = _model.encode_multimodal(
                image=image_data, text=text_data
            )
            score = float(_model.get_matching_scores(joint_embeddings))
            reranked.append((uri, score))

        reranked = sorted(reranked, reverse=True, key=lambda x: x[1])
        uris = [uri for uri, _ in reranked]

    return uris


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
