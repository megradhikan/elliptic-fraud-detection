"""Phase 3/4 training loop, shared by the transductive, inductive, and
edge-shuffle protocols — they differ only in which Data object (which nodes/
edges are visible when) gets passed in, not in how training works.
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.graph_data import build_pyg_data, symmetrize_edge_index
from src.metrics import compute_metrics, save_metrics
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

MODEL_REGISTRY = {"gcn": GCN, "graphsage": GraphSAGE}


def make_model(name: str, in_channels: int, hidden_channels: int = 64, num_layers: int = 2) -> torch.nn.Module:
    cls = MODEL_REGISTRY[name]
    return cls(in_channels=in_channels, hidden_channels=hidden_channels, num_classes=2, num_layers=num_layers)


def _class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    y_train = y[mask]
    n_pos = (y_train == 1).sum().float().clamp(min=1)
    n_neg = (y_train == 0).sum().float().clamp(min=1)
    total = n_pos + n_neg
    # inverse-frequency weighting, same imbalance-handling logic as Phase 2's scale_pos_weight
    return torch.tensor([total / (2 * n_neg), total / (2 * n_pos)])


def train_and_eval(data: Data, model_name: str, num_layers: int = 2, hidden_channels: int = 64,
                    lr: float = 0.01, weight_decay: float = 5e-4, epochs: int = 200,
                    patience: int = 20, device: str | None = None, verbose: bool = True) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = make_model(model_name, in_channels=data.x.size(1), hidden_channels=hidden_channels,
                        num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = _class_weights(data.y, data.train_mask).to(device)

    best_val_aucpr = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_proba = F.softmax(out[data.val_mask], dim=1)[:, 1].cpu().numpy()
            val_y = data.y[data.val_mask].cpu().numpy()
            val_metrics = compute_metrics(val_y, (val_proba >= 0.5).astype(int), val_proba)

        if val_metrics["auc_pr"] > best_val_aucpr:
            best_val_aucpr = val_metrics["auc_pr"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:3d}  loss={loss.item():.4f}  val_f1={val_metrics['f1']:.4f}  "
                  f"val_aucpr={val_metrics['auc_pr']:.4f}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best val AUC-PR={best_val_aucpr:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_proba = F.softmax(out[data.test_mask], dim=1)[:, 1].cpu().numpy()
        test_y = data.y[data.test_mask].cpu().numpy()
    test_preds = (test_proba >= 0.5).astype(int)
    metrics = {
        "model": model_name,
        "num_layers": num_layers,
        "best_val_aucpr": best_val_aucpr,
        **compute_metrics(test_y, test_preds, test_proba),
    }
    return metrics, model


def run_transductive_phase3(n_layer_options=(1, 2, 3)):
    from src.data import load_raw

    raw = load_raw()
    data = build_pyg_data(raw)
    # Standard practice (Weber et al. and most follow-ups): treat the
    # transaction graph as undirected for message passing. This is what
    # lets test-period structure leak backward into training — see
    # graph_data.symmetrize_edge_index and inductive_split.py for the fix.
    data.edge_index = symmetrize_edge_index(data.edge_index)
    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"train/val/test = {int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}")

    all_metrics = []
    for model_name in ("gcn", "graphsage"):
        for num_layers in n_layer_options:
            print(f"\n=== {model_name} (layers={num_layers}), transductive ===")
            metrics, _ = train_and_eval(data, model_name, num_layers=num_layers)
            metrics["protocol"] = "transductive"
            print(json.dumps(metrics, indent=2))
            save_metrics(metrics, RESULTS_DIR / "gnn_metrics.json")
            all_metrics.append(metrics)
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    run_transductive_phase3(tuple(args.layers))
