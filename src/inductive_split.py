"""Phase 4: strict inductive, leakage-free graph construction.

Key idea: filter the edge list so an edge (src -> dst) survives only if
time_step[src] <= time_step[dst]. Since PyG's message-passing convention
aggregates INTO the destination FROM the source, this guarantees every
node's k-hop receptive field is confined to nodes at or before its own time
step, for any k — a node in the training period (the earliest block of time
steps) can therefore only ever aggregate from other training-period nodes,
so a single forward pass over this filtered graph is simultaneously:

  - the strictly inductive TRAINING graph (no test-period node or edge is
    reachable from any node used in the training loss), and
  - the correct EVALUATION graph (a val/test node is scored using only
    neighbors known at or before its own time step, with no forward-looking
    adjacency into later time steps).

This is why train_and_eval() from train_gnn.py can be reused unmodified —
the leakage prevention lives entirely in which edges exist, not in a
separate training/eval code path.
"""

import torch
from torch_geometric.data import Data

from src.graph_data import build_pyg_data


def causal_filter_edges(edge_index: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index[0], edge_index[1]
    keep = time_step[src] <= time_step[dst]
    return edge_index[:, keep]


def build_inductive_data(raw) -> Data:
    data = build_pyg_data(raw)
    n_edges_before = data.edge_index.size(1)
    data.edge_index = causal_filter_edges(data.edge_index, data.time_step)
    n_edges_after = data.edge_index.size(1)
    print(f"Inductive causal filter: kept {n_edges_after}/{n_edges_before} edges "
          f"({100 * n_edges_after / max(n_edges_before, 1):.1f}%)")
    return data


if __name__ == "__main__":
    import json
    from pathlib import Path

    from src.data import load_raw
    from src.metrics import save_metrics
    from src.train_gnn import train_and_eval

    RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

    raw = load_raw()
    data = build_inductive_data(raw)

    all_metrics = []
    for model_name in ("gcn", "graphsage"):
        print(f"\n=== {model_name}, strict inductive ===")
        metrics, _ = train_and_eval(data, model_name, num_layers=2)
        metrics["protocol"] = "inductive"
        print(json.dumps(metrics, indent=2))
        save_metrics(metrics, RESULTS_DIR / "gnn_metrics.json")
        all_metrics.append(metrics)
