# Illicit Bitcoin Wallet Detection

Classifies Bitcoin transactions (graph nodes) as licit or illicit on the
[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set),
and rigorously tests whether transaction-graph structure actually helps once
evaluation matches real deployment conditions — a leakage-free, **inductive**
protocol — rather than the transductive setup most published benchmarks use.

## Why this exists

Nearly every widely-cited GNN result on this dataset evaluates
**transductively**: the encoder sees the full graph, including test-period
edges, at train time — only test *labels* are masked. The literature's
framing is that Elliptic's edges point forward in time (transaction →
transaction), so this leaks future topology into training in a way that
doesn't hold up once you score genuinely new transactions with no access to
future edges — how a real fraud-detection system actually operates.

This project runs the identical baseline and GNN models under both
protocols on one pipeline, and along the way found something that reframes
the analysis: **every edge in the public Elliptic release connects two
transactions in the *same* time step** — 0 of 234,355 edges cross a
time-step boundary (verified directly on the raw CSVs; see
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb), section 3). The graph is
really 49 disconnected components, one per time step, not one graph with
edges pointing forward through time. That makes the literature's
"future-topology leakage" mechanism structurally impossible here regardless
of protocol — which is itself a documented finding, not a shortcut. The
project pivots to the question this dataset can actually answer: a
degree-preserving edge-shuffle ablation testing whether real *within*-time-
step topology carries genuine signal versus random rewiring (see
[`results/leakage_gap_analysis.md`](results/leakage_gap_analysis.md) for the
full writeup), plus a targeted robustness-fix attempt in Phase 5. A negative
result for either is still a reported, valid finding.

## Dataset

[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
— ~203K nodes, ~234K directed edges, 166 features/node (94 local, 72
one-hop-aggregated), 49 time steps. Labels: ~2% illicit, ~21% licit, ~77%
unknown.

> Weber, M. et al. (2019). *Anti-Money Laundering in Bitcoin: Experimenting
> with Graph Convolutional Networks for Financial Forensics.* KDD Workshop
> on Anomaly Detection in Finance.

## The split — and why

Time-based, not random: time steps **1–34 train, 35–39 validation, 40–49
test**. Later time steps are held out entirely, matching how the dataset is
meant to be used and how a real deployment would work — you never train on
the future. See [`src/data.py`](src/data.py) for the exact cut and
[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) for the per-time-step
class balance that justifies it.

## Results

All numbers on the held-out test period (time steps 40–49), illicit class. GCN/GraphSAGE
rows below are 2-layer, matched across protocols for a fair comparison; see
`results/gnn_metrics.json` for the 1/3-layer ablation (GraphSAGE-3-layer transductive
reaches F1=0.612, the single best plain-GNN number found, still short of the baseline).

| Model | Precision | Recall | F1 | AUC-PR |
|---|---|---|---|---|
| Baseline (XGBoost, tuned — 25-trial optuna sweep) | 0.889 | 0.593 | 0.711 | 0.673 |
| Baseline (Random Forest) | 0.975 | 0.553 | 0.706 | 0.665 |
| GCN (transductive) | 0.667 | 0.467 | 0.549 | 0.507 |
| GraphSAGE (transductive) | — | — | 0.292 | — |
| GCN (inductive) | 0.373 | 0.450 | 0.408–0.526* | 0.383–0.447* |
| GraphSAGE (inductive) | 0.172 | 0.736 | 0.314 | 0.401 |
| **GCN + RF confidence-weighted ensemble (Phase 5 fix)** | **0.984** | **0.563** | **0.716** | **0.664** |

_*The inductive GCN has notably high run-to-run variance (F1 std=0.134 across 5 seeds,
see `results/error_analysis.md`) — a limitation worth flagging on its own, not just a
number to average away._

**Headline finding**: on this dataset, no GNN configuration beats the feature-only
baseline on its own (tuned XGBoost, F1=0.711) — and the standard transductive-vs-inductive
"leakage" story from the literature doesn't apply here at all, because **every edge in
the raw Elliptic edgelist connects two nodes in the same time step** (0 of 234,355
cross a time-step boundary, verified directly on the CSVs — see the EDA notebook and
`results/leakage_gap_analysis.md`). The graph is 49 disconnected per-time-step
components, not one graph with forward-pointing edges, so the literature's leakage
mechanism is structurally impossible here regardless of protocol. The more informative
test turned out to be the edge-shuffle ablation: real transaction topology (F1=0.257)
clearly outperforms degree-preserving random rewiring (F1=0.176 ± 0.018) — the graph
structure is carrying genuine signal, it just isn't enough on its own to beat a good
feature-only model. The one fix that helped was the simplest one: a confidence-weighted
ensemble of the GNN and the (untuned) Random Forest baseline (Phase 5) reaches F1=0.716,
edging out every standalone model including the tuned XGBoost baseline above.

Full writeups: [`results/leakage_gap_analysis.md`](results/leakage_gap_analysis.md)
(the central Phase 4 finding), [`results/robustness_attempt.md`](results/robustness_attempt.md)
(Phase 5), [`results/error_analysis.md`](results/error_analysis.md) (Phase 6),
[`results/unknown_node_inference.md`](results/unknown_node_inference.md) (Phase 7).

## Repo structure

```
elliptic-fraud-detection/
  data/raw/              # gitignored — see download_data.py
  notebooks/01_eda.ipynb
  src/
    data.py              # load + merge raw CSVs, time-based split
    graph_data.py        # torch_geometric.data.Data construction
    inductive_split.py   # leakage-free causal edge filter (core contribution)
    edge_shuffle.py       # degree-preserving edge-shuffle ablation
    baseline.py           # XGBoost / Random Forest, feature-only
    train_gnn.py          # shared GCN/GraphSAGE training loop
    robustness_fix.py     # temporal-consistency reg. + confidence ensemble
    error_analysis.py     # Phase 6: per-time-step, FP/FN, PR curve, seed variance
    unknown_node_inference.py  # Phase 7: score the ~77% unlabeled nodes
    metrics.py             # shared precision/recall/F1/AUC-PR helpers
    models/gcn.py
    models/graphsage.py
  results/                # metrics JSON + markdown writeups + figures/
  download_data.py
  requirements.txt
```

## Reproduce

Requires **Python 3.11** (PyTorch Geometric wheels lag newer Python
releases — a 3.14 system interpreter will not resolve `torch-geometric`).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Get the data** (Kaggle auth required — see `download_data.py`'s
   docstring for the one-time API-key setup):
   ```bash
   python download_data.py
   ```
2. **EDA** — class balance, degree distribution, homophily:
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
   ```
3. **Baseline** (feature-only XGBoost + Random Forest):
   ```bash
   python -m src.baseline --sweep
   ```
4. **GNN, transductive** (Phase 3 — replication):
   ```bash
   python -m src.train_gnn
   ```
5. **GNN, strict inductive + edge-shuffle ablation** (Phase 4 — core
   contribution):
   ```bash
   python -m src.inductive_split
   python -m src.edge_shuffle --model gcn
   python -m src.leakage_gap_report
   ```
6. **Robustness fix attempt** (Phase 5, stretch):
   ```bash
   python -m src.robustness_fix
   ```
7. **Error analysis** (Phase 6) and **unknown-node inference** (Phase 7):
   ```bash
   python -m src.error_analysis
   python -m src.unknown_node_inference
   ```

Each step writes its metrics/figures/markdown into `results/`.

## Real-world framing

In production, a false negative here is a missed illicit wallet — funds
keep moving and the compliance exposure compounds. A false positive is a
frozen legitimate account — a real customer locked out, with a support and
trust cost. The precision/recall tradeoff this project measures per model
isn't just a modeling choice, it's a business decision about which failure
mode a deployment is willing to eat more of. That's also why it was worth
verifying the leakage assumption directly on this data rather than taking
the literature's framing on faith — a deployment decision ("is the graph
worth the infrastructure cost of a GNN in production?") should rest on a
protocol that's actually been checked against how the data is structured,
not on an assumption that happened to not hold here.

## Non-goals

No live blockchain API integration, no production deployment, no
hyperparameter-search infrastructure beyond a simple `optuna` sweep.

## Resume bullet

> Investigated a widely cited claim that graph neural networks outperform
> feature-only baselines for Bitcoin fraud detection (Elliptic dataset,
> 203K nodes): built a leakage-free inductive evaluation protocol to test
> it rigorously, discovered along the way that the dataset's edges never
> actually cross time steps (invalidating the literature's assumed leakage
> mechanism — verified directly on the raw data), then re-targeted the
> question to a degree-preserving edge-shuffle ablation showing real
> transaction topology does carry genuine signal (F1=0.257 vs. 0.176±0.018
> for randomly rewired graphs) even though no standalone GNN beat a tuned
> baseline (F1=0.711 vs. 0.612 best standalone GNN); a confidence-weighted
> GNN+baseline ensemble ultimately edged out the standalone baseline too
> (F1=0.716).

## License

Code: MIT (see [LICENSE](LICENSE)). The Elliptic Data Set itself is
distributed by Elliptic/MIT-IBM Watson AI Lab under its own Kaggle license —
see the dataset page for terms.
