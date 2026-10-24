"""Synthetic benchmark with known ground truth.

A document is a set of *topics*.  For each topic t we emit
  * one RULE fragment (the instruction);
  * with probability p_exc, one EXCEPTION fragment, linked to its rule by a
    dependency edge (the two must be sent together);
  * a random number of near-DUPLICATES of the rule (paraphrases obtained by
    substituting a fraction of its words) - the redundancy we want to avoid;
  * a few NOISE fragments that mention the topic loosely.
A query about topic t has gold set {rule_t} plus {exception_t} if it exists.

The exception deliberately shares fewer words with the query than the rule's
duplicates do, which is exactly the situation where relevance-only retrieval
spends the budget on paraphrases and drops the exception.
Costs are whitespace token counts (a proxy for BPE tokens).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Fragment:
    text: str
    topic: int
    kind: str          # rule | exception | duplicate | noise
    origin: int        # id of the fragment it duplicates (itself otherwise)


@dataclass
class Corpus:
    fragments: list[Fragment]
    edges: list[tuple[int, int]]
    rule_of: dict[int, int]
    exception_of: dict[int, int]
    topic_words: list[list[str]]


def make_corpus(n_topics: int, seed: int = 0, p_exc: float = 0.6,
                max_dups: int = 4, noise_per_topic: int = 2) -> Corpus:
    rng = np.random.default_rng(seed)
    filler = [f"f{k}" for k in range(400)]
    topic_words = [[f"t{t}w{k}" for k in range(25)] for t in range(n_topics)]
    exc_words = [f"x{k}" for k in range(60)]   # "unless", "except" ... analogues

    frags: list[Fragment] = []
    edges, rule_of, exception_of = [], {}, {}

    def sentence(words_topic, frac_topic, length, extra=()):
        k = int(round(frac_topic * length))
        w = list(rng.choice(words_topic, size=k)) + list(rng.choice(filler, size=length - k))
        w += list(extra)
        rng.shuffle(w)
        return " ".join(w)

    for t in range(n_topics):
        tw = topic_words[t]
        rule_txt = sentence(tw, 0.55, int(rng.integers(25, 60)))
        rid = len(frags); frags.append(Fragment(rule_txt, t, "rule", rid))
        rule_of[t] = rid
        if rng.random() < p_exc:
            ex = sentence(tw, 0.30, int(rng.integers(20, 45)),
                          extra=rng.choice(exc_words, size=4))
            eid = len(frags); frags.append(Fragment(ex, t, "exception", eid))
            exception_of[t] = eid
            edges.append((rid, eid))
        for _ in range(int(rng.integers(0, max_dups + 1))):
            words = rule_txt.split()
            mask = rng.random(len(words)) < 0.2
            for i in np.flatnonzero(mask):
                words[i] = str(rng.choice(tw if rng.random() < 0.5 else filler))
            frags.append(Fragment(" ".join(words), t, "duplicate", rid))
        for _ in range(noise_per_topic):
            frags.append(Fragment(sentence(tw, 0.15, int(rng.integers(20, 50))),
                                  t, "noise", len(frags)))

    # shuffle fragment order so position carries no information
    perm = rng.permutation(len(frags))
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    frags = [frags[p] for p in perm]
    for f in frags:
        f.origin = int(inv[f.origin])
    edges = [(int(inv[a]), int(inv[b])) for a, b in edges]
    rule_of = {t: int(inv[i]) for t, i in rule_of.items()}
    exception_of = {t: int(inv[i]) for t, i in exception_of.items()}
    return Corpus(frags, edges, rule_of, exception_of, topic_words)


def make_query(corpus: Corpus, topic: int, rng: np.random.Generator) -> tuple[str, set[int]]:
    words = rng.choice(corpus.topic_words[topic], size=8)
    gold = {corpus.rule_of[topic]}
    if topic in corpus.exception_of:
        gold.add(corpus.exception_of[topic])
    return " ".join(words), gold
