"""Reproducible experiments for the midterm write-up.

    python experiments/run.py            # writes results/*.json and results/fig.pdf

Every random choice is driven by an explicit seed, so reruns give identical
numbers (timings excepted).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenopt.pipeline import Index                                  # noqa: E402
from tokenopt.selector import (brute_force, bundle, celf_select,       # noqa: E402
                               dependency_violations, mmr_budget,
                               naive_greedy_select, topk_budget)
from tokenopt.synthetic import make_corpus, make_query                # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEEDS = [0, 1, 2, 3, 4]
METHODS = {
    "top-k": lambda inst: topk_budget(inst),
    "MMR": lambda inst: mmr_budget(inst, lam=0.7),
    "ours": lambda inst: celf_select(inst, m=None),   # pool already applied
}


def wasted_share(sel_global, corpus, costs):
    """Share of selected tokens spent on a 2nd+ copy of the same origin."""
    seen, wasted, total = set(), 0, 0
    for i in sorted(sel_global, key=lambda i: -costs[i]):
        o = corpus.fragments[i].origin
        total += costs[i]
        if o in seen:
            wasted += costs[i]
        seen.add(o)
    return wasted / total if total else 0.0


# --------------------------------------------------------------------------
def quality(budgets=(150, 300, 600), n_topics=60, m=200):
    rows = []
    for seed in SEEDS:
        corpus = make_corpus(n_topics, seed=seed)
        texts = [f.text for f in corpus.fragments]
        idx = Index(texts, corpus.edges)
        full_tokens = int(idx.costs.sum())
        rng = np.random.default_rng(100 + seed)
        queries = [make_query(corpus, t, rng) for t in range(n_topics)]
        for B in budgets:
            for q, gold in queries:
                inst, pool = idx.instance(q, B, m=m)
                rows.append(dict(seed=seed, B=B, method="full", tokens=full_tokens,
                                 complete=1.0, recall=1.0, broken=0.0,
                                 wasted=wasted_share(range(len(texts)), corpus, idx.costs)))
                for name, fn in METHODS.items():
                    sel = fn(inst)
                    g = [pool[i] for i in sel.fragments]
                    hit = len(gold & set(g))
                    rows.append(dict(seed=seed, B=B, method=name, tokens=sel.cost,
                                     complete=float(hit == len(gold)),
                                     recall=hit / len(gold),
                                     broken=float(dependency_violations(inst, sel.fragments) > 0),
                                     wasted=wasted_share(g, corpus, idx.costs)))
    # aggregate: mean over queries, then mean +- sd over seeds
    agg = {}
    keys = sorted({(r["B"], r["method"]) for r in rows})
    for B, meth in keys:
        per_seed = []
        for s in SEEDS:
            rs = [r for r in rows if r["B"] == B and r["method"] == meth and r["seed"] == s]
            per_seed.append({k: np.mean([r[k] for r in rs])
                             for k in ("tokens", "complete", "recall", "broken", "wasted")})
        agg[f"{B}|{meth}"] = {k: (float(np.mean([p[k] for p in per_seed])),
                                  float(np.std([p[k] for p in per_seed])))
                              for k in per_seed[0]}
    return agg


# --------------------------------------------------------------------------
def scaling(topic_counts=(50, 100, 200, 400, 800, 1600), m=200, B=300, reps=20):
    out = []
    for T in topic_counts:
        corpus = make_corpus(T, seed=7)
        texts = [f.text for f in corpus.fragments]
        t0 = time.perf_counter(); idx = Index(texts, corpus.edges); t_index = time.perf_counter() - t0
        rng = np.random.default_rng(7)
        qs = [make_query(corpus, int(t), rng)[0] for t in rng.integers(0, T, size=reps)]
        rec = dict(n=len(texts), index_s=t_index)
        for label, mm in (("pool", m), ("nopool", None)):
            if mm is None and len(texts) > 2500:
                continue
            tb, tl, tn, el, en = [], [], [], [], []
            for q in qs:
                t0 = time.perf_counter(); inst, _ = idx.instance(q, B, m=mm); tb.append(time.perf_counter() - t0)
                t0 = time.perf_counter(); a = celf_select(inst, m=None); tl.append(time.perf_counter() - t0)
                el.append(a.evaluations)
                if len(inst.costs) <= 1200:
                    t0 = time.perf_counter(); b = naive_greedy_select(inst, m=None); tn.append(time.perf_counter() - t0)
                    en.append(b.evaluations)
            rec[label] = dict(build_ms=1e3 * np.median(tb), lazy_ms=1e3 * np.median(tl),
                              lazy_evals=float(np.median(el)),
                              naive_ms=1e3 * np.median(tn) if tn else None,
                              naive_evals=float(np.median(en)) if en else None,
                              pool=len(inst.costs))
        out.append(rec)
        print("scaling", rec, flush=True)
    return out


# --------------------------------------------------------------------------
def approximation(trials=300, m=18):
    """Pools small enough for exhaustive search; ratio f(ours)/f(OPT)."""
    ratios, wins = [], {"greedy": 0, "singleton": 0}
    rng = np.random.default_rng(11)
    for t in range(trials):
        corpus = make_corpus(30, seed=1000 + t)
        idx = Index([f.text for f in corpus.fragments], corpus.edges)
        q, _ = make_query(corpus, int(rng.integers(30)), rng)
        B = int(rng.integers(60, 250))
        inst, _ = idx.instance(q, B, m=m)
        if len(bundle(inst.n, inst.edges)) > 20:
            continue
        opt = brute_force(inst).value
        sel = celf_select(inst, m=None)
        wins[sel.meta["winner"]] += 1
        if opt > 0:
            ratios.append(sel.value / opt)
    r = np.array(ratios)
    return dict(count=len(r), mean=float(r.mean()), min=float(r.min()),
                p5=float(np.percentile(r, 5)), exact_share=float((r > 1 - 1e-9).mean()),
                wins=wins, bound=0.5 * (1 - 1 / np.e))


# --------------------------------------------------------------------------
def consistency(perms=5, n_topics=60, B=300):
    """Same corpus, fragment order permuted: is the selected *set* identical?"""
    corpus = make_corpus(n_topics, seed=3)
    texts = [f.text for f in corpus.fragments]
    rng = np.random.default_rng(5)
    queries = [make_query(corpus, t, rng)[0] for t in range(n_topics)]
    base = []
    idx = Index(texts, corpus.edges)
    for q in queries:
        inst, pool = idx.instance(q, B); base.append({pool[i] for i in celf_select(inst, m=None).fragments})
    jac = []
    for p in range(perms):
        perm = np.random.default_rng(50 + p).permutation(len(texts))     # new id k -> old perm[k]
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        idx2 = Index([texts[i] for i in perm], [(int(inv[a]), int(inv[b])) for a, b in corpus.edges])
        for q, S0 in zip(queries, base):
            inst, pool = idx2.instance(q, B)
            S = {int(perm[pool[i]]) for i in celf_select(inst, m=None).fragments}
            jac.append(len(S & S0) / len(S | S0) if S | S0 else 1.0)
    return dict(mean_jaccard=float(np.mean(jac)), identical=float(np.mean(np.array(jac) == 1.0)))


def figure(agg, scal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif"})
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.1))
    budgets = [150, 300, 600]
    style = {"top-k": ("#9a9a9a", "s"), "MMR": ("#5b8db8", "^"), "ours": ("#c0392b", "o")}
    for meth, (c, mk) in style.items():
        y = [100 * agg[f"{B}|{meth}"]["complete"][0] for B in budgets]
        e = [100 * agg[f"{B}|{meth}"]["complete"][1] for B in budgets]
        ax[0].errorbar(budgets, y, yerr=e, color=c, marker=mk, ms=4, lw=1.2, capsize=2, label=meth)
    ax[0].set_xlabel("token budget $B$"); ax[0].set_ylabel("complete answers (%)")
    ax[0].set_xticks(budgets); ax[0].set_ylim(0, 105); ax[0].legend(frameon=False, loc="lower right")
    ax[0].spines[["top", "right"]].set_visible(False)
    n = [r["n"] for r in scal]
    ax[1].plot(n, [r["pool"]["build_ms"] + r["pool"]["lazy_ms"] for r in scal], "o-", color="#c0392b", ms=4, lw=1.2, label="pool $m=200$")
    nn = [r for r in scal if "nopool" in r]
    ax[1].plot([r["n"] for r in nn], [r["nopool"]["build_ms"] + r["nopool"]["lazy_ms"] for r in nn], "s--", color="#555", ms=4, lw=1.2, label="no pool ($m=n$)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("fragments $n$"); ax[1].set_ylabel("ms per query (median)")
    ax[1].legend(frameon=False); ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = {}
    res["quality"] = quality(); print("quality done", flush=True)
    res["approximation"] = approximation(); print("approx", res["approximation"], flush=True)
    res["consistency"] = consistency(); print("consistency", res["consistency"], flush=True)
    res["scaling"] = scaling()
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    figure(res["quality"], res["scaling"])
    print("written", OUT)
