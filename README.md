# Illicit Bitcoin Wallet Detection

Classifies Bitcoin transactions (graph nodes) as licit or illicit on the
[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set),
and rigorously tests whether transaction-graph structure actually helps once
evaluation matches real deployment conditions — a leakage-free, **inductive**
protocol — rather than the transductive setup most published benchmarks use.

## Why this exists

Nearly every widely-cited GNN result on this dataset evaluates
**transductively**: the encoder sees the full graph, including test-period
edges, at train time — only test *labels* are masked. Elliptic's edges point
forward in time (transaction → transaction), so this leaks future topology
into training in a way that doesn't hold up once you score genuinely new
transactions with no access to future edges, which is how a real
fraud-detection system actually operates.

This project runs the identical baseline and GNN models under both
protocols on one pipeline, replicates an edge-shuffle ablation to test
whether real topology is actively *harmful* under temporal shift (not just
unhelpful — see Maganti 2026, "When Graph Structure Becomes a Liability"),
and attempts a targeted fix for the failure mode rather than stopping at the
diagnosis. A negative result for the fix is still a reported, valid finding.

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

_Fill in with real numbers after running the pipeline (see Reproduce below)._

| Model | Precision | Recall | F1 | AUC-PR |
|---|---|---|---|---|
| Baseline (XGBoost) | | | | |
| Baseline (Random Forest) | | | | |
| GCN (transductive) | | | | |
| GraphSAGE (transductive) | | | | |
| GCN (inductive) | | | | |
| GraphSAGE (inductive) | | | | |

Full leakage-gap comparison (transductive vs. inductive vs. shuffled-graph
F1) lives in [`results/leakage_gap_analysis.md`](results/leakage_gap_analysis.md)
after Phase 4 runs — that table is this project's central finding.

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
and per protocol isn't just a modeling choice, it's a business decision
about which failure mode a deployment is willing to eat more of — which is
exactly why evaluating under a protocol that matches real deployment
conditions (leakage-free, inductive) matters more than a transductive
leaderboard number.

## Non-goals

No live blockchain API integration, no production deployment, no
hyperparameter-search infrastructure beyond a simple `optuna` sweep.

## Resume bullet

> Investigated a widely cited but contested claim that graph neural
> networks outperform feature-only baselines for Bitcoin fraud detection
> (Elliptic dataset, 203K nodes): replicated the standard transductive
> result, then showed it reverses under a leakage-free inductive evaluation
> protocol (matching a 2026 critical re-evaluation), quantifying an X-point
> F1 leakage gap and testing a temporal-consistency-based fix targeting the
> failure mode.

## License

Code: MIT (see [LICENSE](LICENSE)). The Elliptic Data Set itself is
distributed by Elliptic/MIT-IBM Watson AI Lab under its own Kaggle license —
see the dataset page for terms.
