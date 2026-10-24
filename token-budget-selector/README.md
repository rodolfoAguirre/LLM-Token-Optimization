# Token-budget context selector — TC2038 Midterm 1, Equipo 5

Selects which document fragments to send to an LLM so that the context fits a
token budget `B`, avoids sending paraphrases of the same content, and never
splits fragments that must travel together (e.g. an instruction and its
exception).

**Status: midterm progress (prototype).** The algorithm, baselines, tests and a
synthetic benchmark with known ground truth are implemented. End-to-end
evaluation with a real LLM and real documents is the next stage.

## Layout

| Path | Contents |
|---|---|
| `tokenopt/selector.py` | Union-find bundling, facility-location objective, lazy greedy (CELF) + best-singleton rule, naive greedy, baselines (top-k, MMR, full context), exact brute force |
| `tokenopt/embed.py` | Deterministic hashed TF-IDF embedding (stand-in for a neural model) |
| `tokenopt/pipeline.py` | Document index and per-query candidate pool |
| `tokenopt/synthetic.py` | Benchmark generator: rules, exceptions, near-duplicates, noise |
| `tests/test_selector.py` | Monotonicity/submodularity, feasibility, lazy = naive, bound vs brute force |
| `experiments/run.py` | All experiments in the write-up → `results/results.json`, `results/fig.pdf` |
| `doc/onepager.tex` | Midterm write-up |

## Run

```bash
pip install numpy matplotlib        # Python >= 3.10
python tests/test_selector.py       # or: python -m pytest -q
python experiments/run.py           # ~1 min; seeds are fixed
```

## Minimal use

```python
from tokenopt.pipeline import Index
from tokenopt.selector import celf_select

idx = Index(fragments, edges)              # edges: pairs that must go together
inst, pool = idx.instance(query, budget=300, m=200)
sel = celf_select(inst, m=None)
context = [fragments[pool[i]] for i in sel.fragments]
```
