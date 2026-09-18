"""Phase 4: edge-shuffle ablation.

Randomly rewires the inductive graph's edges with a directed configuration
model (preserves each node's in-degree and out-degree, so any performance
difference is attributable to *which* nodes are connected, not how many).
The causal time filter is re-applied after shuffling so the ablation isolates
"does real topology help" without reintroducing leakage as a confound — the
comparison to the real inductive graph stays apples-to-apples.

If Maganti (2026)'s finding replicates, the shuffled graph should score
comparably to, or better than, the real one.
"""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from src.data import load_raw
from src.inductive_split import build_inductive_data, causal_filter_edges
from src.metrics import save_metrics
from src.train_gnn import train_and_eval

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def shuffle_edges(edge_index: torch.Tensor, num_nodes: int, seed: int) -> torch.Tensor:
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    out_deg = np.zeros(num_nodes, dtype=int)
    in_deg = np.zeros(num_nodes, dtype=int)
    for s, d in zip(src, dst):
        out_deg[s] += 1
        in_deg[d] += 1

    g = nx.directed_configuration_model(out_deg.tolist(), in_deg.tolist(), seed=seed)
    g = nx.DiGraph(g)  # collapse parallel edges, drop self loops implicitly handled below
    g.remove_edges_from(nx.selfloop_edges(g))

    new_src = np.fromiter((u for u, v in g.edges()), dtype=np.int64, count=g.number_of_edges())
    new_dst = np.fromiter((v for u, v in g.edges()), dtype=np.int64, count=g.number_of_edges())
    return torch.tensor(np.stack([new_src, new_dst]))


def run_ablation(model_name: str = "gcn", num_layers: int = 2, seeds: tuple[int, ...] = tuple(range(5))) -> dict:
    raw = load_raw()
    data = build_inductive_data(raw)  # real, causally-filtered inductive graph

    real_metrics, _ = train_and_eval(data, model_name, num_layers=num_layers, verbose=False)
    print(f"Real graph: F1={real_metrics['f1']:.4f} AUC-PR={real_metrics['auc_pr']:.4f}")

    shuffled_f1, shuffled_aucpr = [], []
    for seed in seeds:
        shuffled_edge_index = shuffle_edges(data.edge_index, data.num_nodes, seed=seed)
        shuffled_edge_index = causal_filter_edges(shuffled_edge_index, data.time_step)

        shuffled_data = data.clone()
        shuffled_data.edge_index = shuffled_edge_index

        metrics, _ = train_and_eval(shuffled_data, model_name, num_layers=num_layers, verbose=False)
        print(f"  seed={seed}: F1={metrics['f1']:.4f} AUC-PR={metrics['auc_pr']:.4f}")
        shuffled_f1.append(metrics["f1"])
        shuffled_aucpr.append(metrics["auc_pr"])

    result = {
        "model": model_name,
        "num_layers": num_layers,
        "real_graph": {"f1": real_metrics["f1"], "auc_pr": real_metrics["auc_pr"]},
        "shuffled_graph": {
            "f1_mean": float(np.mean(shuffled_f1)), "f1_std": float(np.std(shuffled_f1)),
            "auc_pr_mean": float(np.mean(shuffled_aucpr)), "auc_pr_std": float(np.std(shuffled_aucpr)),
            "seeds": list(seeds),
        },
    }
    save_metrics(result, RESULTS_DIR / "edge_shuffle_ablation.json")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gcn", choices=["gcn", "graphsage"])
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    run_ablation(args.model, args.num_layers, tuple(args.seeds))
