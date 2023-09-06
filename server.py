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

    if not os.path.exists(os.path.join(dir, "v2.3.0")):
        os.mkdir(os.path.join(dir, "v2.3.0"))
    index_path = os.path.join(dir, "v2.3.0", "images.uform-vl-multilingual-v2.usearch")
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
    name: _open_dataset(os.path.join("/data", name))
    for name in (
        "unsplash-25k",
        "cc-3m",
        # "laion-4m",
    )
}


def find_vector(
    vector: np.ndarray,
    count: int = 10,
    include_unsplash: bool = True,
    include_cc: bool = True,
    include_laion: bool = True,
) -> List[str]:
    uris_and_distances = []
    vector = vector.flatten()

    if include_unsplash:
        dataset_object = _datasets["unsplash-25k"]
        matches: Matches = dataset_object.index.search(vector, count)
        ids = matches.keys.flatten()
        uris: List[str] = [str(dataset_object.uris[id]) for id in ids]
        distances = matches.distances.flatten()
        uris_and_distances.extend(list(zip(uris, distances)))

    if include_cc:
        dataset_object = _datasets["cc-3m"]
        matches: Matches = dataset_object.index.search(vector, count)
        ids = matches.keys.flatten()
        uris: List[str] = [str(dataset_object.uris[id]) for id in ids]
        distances = matches.distances.flatten()
        uris_and_distances.extend(list(zip(uris, distances)))

    if include_laion:
        dataset_object = _datasets["laion-4m"]
        matches: Matches = dataset_object.index.search(vector, count)
        ids = matches.keys.flatten()
        uris: List[str] = [str(dataset_object.uris[id]) for id in ids]
        distances = matches.distances.flatten()
        uris_and_distances.extend(list(zip(uris, distances)))

    uris_and_distances.sort(key=lambda x: x[1])
    return [uri for uri, _ in uris_and_distances]


def size(
    include_unsplash: bool = True,
    include_cc: bool = True,
    include_laion: bool = True,
) -> int:
    """Number of entries in the index"""
    total = 0
    if include_unsplash:
        total += len(_datasets["unsplash-25k"].index)
    if include_cc:
        total += len(_datasets["cc-3m"].index)
    if include_laion:
        total += len(_datasets["laion-4m"].index)
    return total


def sample_images(
    count: int = 10,
    include_unsplash: bool = True,
    include_cc: bool = True,
    include_laion: bool = True,
) -> List[str]:
    #
    candidates_across_all = []
    probabilities_across_all = []

    if include_unsplash:
        dataset = _datasets["unsplash-25k"]
        size = len(dataset.index)
        indexes = np.random.randint(0, size, count)
        images = [str(dataset.uris[i]) for i in indexes]
        candidates_across_all.extend(images)
        probabilities_across_all.extend([1 / size] * count)

    if include_cc:
        dataset = _datasets["cc-3m"]
        size = len(dataset.index)
        indexes = np.random.randint(0, size, count)
        images = [str(dataset.uris[i]) for i in indexes]
        candidates_across_all.extend(images)
        probabilities_across_all.extend([1 / size] * count)

    if include_laion:
        dataset = _datasets["laion-4m"]
        size = len(dataset.index)
        indexes = np.random.randint(0, size, count)
        images = [str(dataset.uris[i]) for i in indexes]
        candidates_across_all.extend(images)
        probabilities_across_all.extend([1 / size] * count)

    p = np.array(probabilities_across_all)
    p = p / p.sum()
    return np.random.choice(
        candidates_across_all,
        size=count,
        p=p,
        replace=False,
    )


def find_with_vector(dataset: str, query: str, count: int) -> List[str]:
    """For the given `query` ASCII vector returns the URIs of the most similar images"""
    return find_vector(dataset, _ascii_to_vector(query), count)


def find_with_text(
    query: str,
    count: int,
    rerank: bool = False,
    include_unsplash: bool = True,
    include_cc: bool = True,
    include_laion: bool = True,
) -> List[str]:
    """For the given `query` string returns the URIs of the most similar images"""
    if query is None or len(query) == 0:
        return sample_images(
            count,
            include_unsplash=include_unsplash,
            include_cc=include_cc,
            include_laion=include_laion,
        )

    text_data = _model.preprocess_text(query)
    text_embedding = _model.encode_text(text_data).detach().numpy()
    uris = find_vector(
        text_embedding,
        count,
        include_unsplash=include_unsplash,
        include_cc=include_cc,
        include_laion=include_laion,
    )

    if rerank:
        reranked = []
        for uri in uris:
            data: str = re.sub("^data:image/.+;base64,", "", uri)
            image = Image.open(io.BytesIO(base64.b64decode(data)))
            image_data = _model.preprocess_image(image)
            joint_embeddings = _model.encode_multimodal(
                image=image_data,
                text=text_data,
            )
            score = float(_model.get_matching_scores(joint_embeddings))
            reranked.append((uri, score))

        reranked = sorted(reranked, reverse=True, key=lambda x: x[1])
        uris = [uri for uri, _ in reranked]

    return uris[:count]


def find_with_image(
    query: Image,
    count: int,
    include_unsplash: bool = True,
    include_cc: bool = True,
    include_laion: bool = True,
) -> List[str]:
    """For the given `query` image returns the URIs of the most similar images"""
    image_data = _model.preprocess_image(query)
    image_embedding = _model.encode_image(image_data).detach().numpy()
    return find_vector(
        image_embedding,
        count,
        include_unsplash=include_unsplash,
        include_cc=include_cc,
        include_laion=include_laion,
    )[:count]


if __name__ == "__main__":
    server = Server()
    server(find_with_vector)
    server(find_with_text)
    server(find_with_image)
    server(size)
    server.run()
