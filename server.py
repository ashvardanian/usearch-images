from typing import List

import numpy as np
from PIL.Image import Image

from ucall.posix import Server
from usearch.index import Index, MetricKind, Matches
from usearch.io import load_matrix
from usearch.server import _ascii_to_vector
from uform import get_model


model = get_model('unum-cloud/uform-vl-english')
vectors = load_matrix('images.fbin')
ndim = vectors.shape[1]
index = Index(ndim=ndim, metric=MetricKind.Cos, path='images.usearch')
uris = open('images.txt', 'r').read().splitlines()


def find_vector(vector: np.ndarray, count: int = 10) -> List[str]:
    assert len(vector) == ndim, 'Wrong number of dimensions in query matrix'
    matches: Matches = index.search(vector, count)
    ids: np.ndarray = matches.labels.flatten()
    return [uris[id] for id in ids]


server = Server()


@server
def find_with_vector(query: str, count: int) -> List[str]:
    """For the given `query` ASCII vector returns the URIs of the most similar images"""
    return find_vector(_ascii_to_vector(query), count)


@server
def find_with_text(query: str, count: int) -> List[str]:
    """For the given `query` string returns the URIs of the most similar images"""
    text_data = model.preprocess_text(query)
    text_embedding = model.encode_text(text_data)
    return find_vector(text_embedding, count)


@server
def find_with_image(query: Image, count: int) -> List[str]:
    """For the given `query` image returns the URIs of the most similar images"""
    image_data = model.preprocess_image(query)
    image_embedding = model.encode_image(image_data)
    return find_vector(image_embedding, count)


server.run()
