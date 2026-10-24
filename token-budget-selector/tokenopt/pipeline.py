"""Glue: document -> index -> per-query Instance restricted to a candidate pool.

Per query the work is
    relevance of all n fragments           O(n d)
    pool of >= m fragments (whole blocks)  O(n log n)
    similarities inside the pool           O(m^2 d)
so the n x n similarity matrix is never materialised.
"""
from __future__ import annotations

import numpy as np

from .embed import HashedTfidf, tokenize
from .selector import Instance, bundle


class Index:
    def __init__(self, texts: list[str], edges, dim: int = 4096):
        self.texts = texts
        self.edges = list(edges)
        self.vec = HashedTfidf(dim).fit(texts)
        self.X = self.vec.transform(texts)                       # (n, d)
        self.costs = np.array([max(1, len(tokenize(t))) for t in texts], dtype=int)
        self.blocks = bundle(len(texts), self.edges)

    def instance(self, query: str, budget: int, m: int | None = 200):
        """Returns (Instance over the pool, list mapping pool row -> fragment id)."""
        q = self.vec.transform([query])[0]
        rel = np.clip(self.X @ q, 0.0, 1.0)
        if m is None or m >= len(self.texts):
            pool = list(range(len(self.texts)))
        else:
            ranked = sorted(self.blocks, key=lambda b: (-rel[b].max(), b[0]))
            pool, size = [], 0
            for b in ranked:
                if size >= m:
                    break
                pool.extend(b); size += len(b)
            pool.sort()
        pos = {g: k for k, g in enumerate(pool)}
        Xp = self.X[pool]
        sim = np.clip(Xp @ Xp.T, 0.0, 1.0)
        np.fill_diagonal(sim, 1.0)
        edges = [(pos[a], pos[b]) for a, b in self.edges if a in pos and b in pos]
        inst = Instance(self.costs[pool].copy(), rel[pool].copy(), sim, edges, budget)
        return inst, pool
