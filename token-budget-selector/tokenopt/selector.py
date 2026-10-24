"""Budgeted, redundancy-aware, dependency-closed context selection.

Problem (see doc/onepager.tex, Section 1):
    given fragments D = {0..n-1} with token costs c_i, query relevances r_i,
    pairwise similarities s_ij in [0, 1], a symmetric dependency relation E
    and a token budget B, choose S subset of D maximising

        f(S) = sum_i r_i * max_{j in S} s_ij          (facility location)

    subject to  sum_{i in S} c_i <= B   and   S closed under E.

Dependencies are removed by contracting connected components of (D, E) into
blocks (union-find); the problem becomes budgeted monotone submodular
maximisation over blocks, solved with the cost-benefit greedy + best-singleton
rule of Khuller, Moss & Naor (1999), evaluated lazily (CELF, Leskovec et al.
2007).  Guarantee: f(S) >= (1/2)(1 - 1/e) * OPT.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------
# Instance and result containers
# --------------------------------------------------------------------------
@dataclass
class Instance:
    costs: np.ndarray            # (n,) positive ints: tokens per fragment
    rel: np.ndarray              # (n,) relevance to the query, in [0, 1]
    sim: np.ndarray              # (n, n) similarity, in [0, 1], sim[i, i] = 1
    edges: list[tuple[int, int]] # symmetric "must be included together"
    budget: int                  # B

    @property
    def n(self) -> int:
        return len(self.costs)


@dataclass
class Selection:
    fragments: list[int]
    value: float
    cost: int
    evaluations: int = 0          # number of marginal-gain evaluations
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Dependency closure: union-find
# --------------------------------------------------------------------------
class UnionFind:
    """Disjoint sets with union by rank and path compression.
    m operations on n elements cost O(m * alpha(n))."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def bundle(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Connected components of (D, E), each sorted, ordered by smallest id."""
    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(uf.find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


# --------------------------------------------------------------------------
# Objective
# --------------------------------------------------------------------------
class FacilityLocation:
    """f(S) = sum_i w_i * max_{j in S} sim[i, j], with f(empty) = 0.

    `cur[i]` caches max_{j in S} sim[i, j]; a marginal gain for a block b is
    sum_i w_i * max(0, max_{j in b} sim[i, j] - cur[i]),  O(|ground| * |b|).
    Monotone and submodular whenever sim >= 0 and w >= 0."""

    def __init__(self, sim: np.ndarray, weights: np.ndarray):
        if (sim < 0).any() or (weights < 0).any():
            raise ValueError("facility location needs non-negative sim and weights")
        self.sim = sim
        self.w = weights
        self.cur = np.zeros(sim.shape[0])

    def gain(self, block: list[int]) -> float:
        cover = self.sim[:, block].max(axis=1)
        return float(self.w @ np.maximum(0.0, cover - self.cur))

    def add(self, block: list[int]) -> None:
        self.cur = np.maximum(self.cur, self.sim[:, block].max(axis=1))

    @property
    def value(self) -> float:
        return float(self.w @ self.cur)


def objective(inst: Instance, S: list[int], ground: np.ndarray | None = None) -> float:
    """f(S) evaluated on the ground set `ground` (default: all fragments)."""
    if not S:
        return 0.0
    g = np.arange(inst.n) if ground is None else ground
    return float(inst.rel[g] @ inst.sim[np.ix_(g, S)].max(axis=1))


# --------------------------------------------------------------------------
# Candidate pool (keeps closure: whole blocks only)
# --------------------------------------------------------------------------
def candidate_pool(inst: Instance, blocks: list[list[int]], m: int | None) -> list[list[int]]:
    """Keep the most relevant blocks until they hold >= m fragments."""
    if m is None or m >= inst.n:
        return blocks
    ranked = sorted(blocks, key=lambda b: (-inst.rel[b].max(), b[0]))
    pool, size = [], 0
    for b in ranked:
        if size >= m:
            break
        pool.append(b)
        size += len(b)
    return sorted(pool, key=lambda b: b[0])


# --------------------------------------------------------------------------
# Proposed algorithm
# --------------------------------------------------------------------------
def _prepare(inst: Instance, m: int | None):
    blocks = candidate_pool(inst, bundle(inst.n, inst.edges), m)
    ground = np.array(sorted(i for b in blocks for i in b))
    pos = {g: k for k, g in enumerate(ground)}           # fragment -> row
    local = [[pos[i] for i in b] for b in blocks]
    bcost = [int(inst.costs[b].sum()) for b in blocks]
    sim = inst.sim[np.ix_(ground, ground)]
    return blocks, ground, local, bcost, sim, inst.rel[ground]


def _finish(inst, blocks, ground, chosen, greedy_val, single, evals, how):
    best_k, best_val = single
    if best_k is not None and best_val > greedy_val:
        picked, value, used = [best_k], best_val, "singleton"
    else:
        picked, value, used = chosen, greedy_val, "greedy"
    frags = sorted(i for k in picked for i in blocks[k])
    return Selection(frags, value, int(inst.costs[frags].sum()) if frags else 0,
                     evals, {"winner": used, "pool_size": len(ground),
                             "blocks": len(blocks), "variant": how})


def celf_select(inst: Instance, m: int | None = 200, min_ratio: float = 0.0) -> Selection:
    """Cost-benefit greedy with lazy evaluation (CELF), then compare with the
    best feasible single block and return the better of the two.

    Invariants of the main loop:
      (I1) cost(S) <= B and S is a union of whole blocks (closed under E);
      (I2) every key stored in the heap is >= the block's current true ratio
           (f(S + b) - f(S)) / c_b, because by submodularity marginal gains
           never increase as S grows.  Hence a popped key that is *fresh*
           (computed for the current S) is a true maximum.
    """
    blocks, ground, local, bcost, sim, w = _prepare(inst, m)
    F = FacilityLocation(sim, w)
    evals, remaining, chosen, step = 0, inst.budget, [], 0
    heap, single = [], (None, -1.0)
    for k, b in enumerate(local):
        if bcost[k] <= inst.budget:
            g = F.gain(b); evals += 1                     # = f({b})
            if g > single[1]:
                single = (k, g)
            heapq.heappush(heap, (-g / bcost[k], k, 0))   # (key, tiebreak, stamp)
    while heap:
        negkey, k, stamp = heapq.heappop(heap)
        if bcost[k] > remaining:          # remaining budget only shrinks: drop
            continue
        if stamp == step:                 # fresh -> true maximum by (I2)
            if -negkey <= min_ratio:      # min_ratio > 0: stop when a token buys too little
                break
            F.add(local[k]); chosen.append(k)
            remaining -= bcost[k]; step += 1
        else:                             # stale -> recompute, push back
            g = F.gain(local[k]); evals += 1
            heapq.heappush(heap, (-g / bcost[k], k, step))
    return _finish(inst, blocks, ground, chosen, F.value, single, evals, "lazy")


def naive_greedy_select(inst: Instance, m: int | None = 200) -> Selection:
    """Same rule, recomputing every gain at every step (reference for CELF)."""
    blocks, ground, local, bcost, sim, w = _prepare(inst, m)
    F = FacilityLocation(sim, w)
    evals, remaining, chosen = 0, inst.budget, []
    alive = [k for k in range(len(local)) if bcost[k] <= inst.budget]
    single = (None, -1.0)
    for k in alive:
        g = F.gain(local[k]); evals += 1
        if g > single[1]:
            single = (k, g)
    while True:
        alive = [k for k in alive if bcost[k] <= remaining and k not in chosen]
        best = None
        for k in alive:
            r = F.gain(local[k]) / bcost[k]; evals += 1
            if best is None or r > best[0] or (r == best[0] and k < best[1]):
                best = (r, k)
        if best is None or best[0] <= 0:
            break
        F.add(local[best[1]]); chosen.append(best[1])
        remaining -= bcost[best[1]]
    return _finish(inst, blocks, ground, chosen, F.value, single, evals, "naive")


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def full_context(inst: Instance) -> Selection:
    """Unoptimised baseline: send everything (ignores B)."""
    S = list(range(inst.n))
    return Selection(S, objective(inst, S), int(inst.costs.sum()))


def topk_budget(inst: Instance) -> Selection:
    """Naive retrieval: most relevant first, while it fits."""
    S, used = [], 0
    for i in sorted(range(inst.n), key=lambda i: (-inst.rel[i], i)):
        if used + inst.costs[i] <= inst.budget:
            S.append(i); used += int(inst.costs[i])
    return Selection(sorted(S), objective(inst, S), used)


def mmr_budget(inst: Instance, lam: float = 0.7) -> Selection:
    """Maximal Marginal Relevance (Carbonell & Goldstein 1998), budgeted."""
    S, used = [], 0
    maxsim = np.zeros(inst.n)
    left = set(range(inst.n))
    while left:
        score = {i: lam * inst.rel[i] - (1 - lam) * maxsim[i]
                 for i in left if used + inst.costs[i] <= inst.budget}
        if not score:
            break
        i = max(score, key=lambda j: (score[j], -j))
        S.append(i); used += int(inst.costs[i]); left.discard(i)
        maxsim = np.maximum(maxsim, inst.sim[:, i])
    return Selection(sorted(S), objective(inst, S), used)


# --------------------------------------------------------------------------
# Exact solver for small instances (to measure the empirical ratio)
# --------------------------------------------------------------------------
def brute_force(inst: Instance) -> Selection:
    blocks = bundle(inst.n, inst.edges)
    if len(blocks) > 22:
        raise ValueError("too many blocks for exhaustive search")
    bcost = [int(inst.costs[b].sum()) for b in blocks]
    best, best_val = [], 0.0
    for r in range(1, len(blocks) + 1):
        for combo in itertools.combinations(range(len(blocks)), r):
            if sum(bcost[k] for k in combo) > inst.budget:
                continue
            S = [i for k in combo for i in blocks[k]]
            v = objective(inst, S)
            if v > best_val:
                best, best_val = sorted(S), v
    return Selection(best, best_val, int(inst.costs[best].sum()) if best else 0)


def dependency_violations(inst: Instance, S: list[int]) -> int:
    s = set(S)
    return sum((a in s) != (b in s) for a, b in inst.edges)
