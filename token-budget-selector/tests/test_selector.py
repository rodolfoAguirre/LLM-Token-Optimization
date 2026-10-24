"""Correctness checks.  Run:  python -m pytest -q   (or  python tests/test_selector.py)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenopt.selector import (Instance, bundle, brute_force, celf_select,  # noqa: E402
                               dependency_violations, naive_greedy_select, objective)

RATIO = 0.5 * (1 - 1 / np.e)          # Khuller-Moss-Naor guarantee


def random_instance(rng, n=12, budget=None, p_edge=0.12):
    X = rng.random((n, 6)) ** 3
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    sim = np.clip(X @ X.T, 0, 1); np.fill_diagonal(sim, 1.0)
    rel = rng.random(n)
    costs = rng.integers(5, 60, size=n)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p_edge]
    B = budget if budget is not None else int(costs.sum() * rng.uniform(0.2, 0.5))
    return Instance(costs, rel, sim, edges, B)


def test_bundle_components():
    assert bundle(6, [(0, 1), (1, 2), (4, 5)]) == [[0, 1, 2], [3], [4, 5]]


def test_objective_monotone_submodular():
    rng = np.random.default_rng(1)
    inst = random_instance(rng, n=10)
    for _ in range(300):
        A = list(np.flatnonzero(rng.random(10) < 0.3))
        extra = list(np.flatnonzero(rng.random(10) < 0.3))
        Bset = sorted(set(A) | set(extra))              # A subset of B
        x = int(rng.integers(10))
        fA, fB = objective(inst, A), objective(inst, Bset)
        assert fB >= fA - 1e-12                                     # monotone
        gA = objective(inst, sorted(set(A) | {x})) - fA
        gB = objective(inst, sorted(set(Bset) | {x})) - fB
        assert gA >= gB - 1e-12                                     # submodular


def test_feasible_and_closed():
    rng = np.random.default_rng(2)
    for _ in range(200):
        inst = random_instance(rng, n=int(rng.integers(5, 30)))
        sel = celf_select(inst, m=None)
        assert sel.cost <= inst.budget
        assert dependency_violations(inst, sel.fragments) == 0


def test_lazy_matches_naive():
    rng = np.random.default_rng(3)
    for _ in range(200):
        inst = random_instance(rng, n=int(rng.integers(5, 40)))
        a, b = celf_select(inst, m=None), naive_greedy_select(inst, m=None)
        assert abs(a.value - b.value) < 1e-9
        assert a.evaluations <= b.evaluations


def test_approximation_bound_against_brute_force():
    rng = np.random.default_rng(4)
    worst = 1.0
    for _ in range(150):
        inst = random_instance(rng, n=int(rng.integers(6, 14)))
        opt = brute_force(inst).value
        if opt == 0:
            continue
        r = celf_select(inst, m=None).value / opt
        worst = min(worst, r)
        assert r >= RATIO - 1e-9
    print(f"worst empirical ratio over random instances: {worst:.3f}")


def test_singleton_rule_needed():
    # One tiny item with a great ratio vs one big item worth far more.
    sim = np.eye(2)
    inst = Instance(np.array([1, 100]), np.array([0.05, 1.0]), sim, [], budget=100)
    sel = celf_select(inst, m=None)
    assert sel.fragments == [1] and sel.meta["winner"] == "singleton"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
