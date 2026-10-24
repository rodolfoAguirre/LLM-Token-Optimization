"""Deterministic, dependency-free text embedding used by the prototype.

A hashed TF-IDF vectoriser: every word is mapped to one of `dim` buckets by a
stable hash, term frequency is sublinearly scaled (1 + log tf), multiplied by
the inverse document frequency computed over the corpus, and each vector is
L2-normalised.  Because every component is non-negative, cosine similarity
between two vectors lies in [0, 1], which the facility-location objective in
`selector.py` requires (it must be monotone, hence similarities must be >= 0).

This is a stand-in: the algorithm only consumes a similarity matrix, so a
neural sentence-embedding model can replace this module without touching the
selector.
"""
from __future__ import annotations

import re
import zlib

import numpy as np

_WORD = re.compile(r"[a-z0-9áéíóúñü]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bucket(word: str, dim: int) -> int:
    # crc32 is stable across runs and platforms, unlike Python's hash().
    return zlib.crc32(word.encode("utf-8")) % dim


class HashedTfidf:
    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.idf: np.ndarray | None = None

    def _tf(self, texts: list[str]) -> np.ndarray:
        X = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for w in tokenize(text):
                X[row, _bucket(w, self.dim)] += 1.0
        nz = X > 0
        X[nz] = 1.0 + np.log(X[nz])
        return X

    def fit(self, corpus: list[str]) -> "HashedTfidf":
        tf = self._tf(corpus)
        df = (tf > 0).sum(axis=0)
        n = len(corpus)
        self.idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self.idf is None:
            raise RuntimeError("call fit() first")
        X = self._tf(texts) * self.idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms
