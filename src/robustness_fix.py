"""Phase 5 (stretch): two targeted fixes for the inductive-setting gap.

1. Temporal-consistency regularization: penalize large embedding shifts
   between structurally similar nodes in adjacent time steps, so the model
   relies less on topology specific to one time window (inspired by
   STC-MixHop).
2. Confidence-weighted ensemble of the inductive GNN and the feature-only
   baseline, mixing coefficient chosen on the validation set.

Either result — fixed the gap or didn't — is a valid, reportable outcome; the
point is that a targeted fix was attempted and measured against Phase 4's
numbers, not assumed to work.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.baseline import get_splits, train_rf
from src.data import load_raw
from src.inductive_split import build_inductive_data
from src.metrics import compute_metrics, save_metrics
from src.train_gnn import _class_weights, make_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def temporal_consistency_loss(embeddings: torch.Tensor, edge_index: torch.Tensor,
                               time_step: torch.Tensor) -> torch.Tensor:
    """Penalize embedding distance between connected nodes in ADJACENT time
    steps (|dt| == 1) — encourages smooth representations across time rather
    than ones that key off a single window's specific topology."""
    src, dst = edge_index[0], edge_index[1]
    adjacent = (time_step[dst] - time_step[src]).abs() == 1
    if adjacent.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    diff = embeddings[src[adjacent]] - embeddings[dst[adjacent]]
    return diff.pow(2).sum(dim=1).mean()


def train_with_temporal_consistency(data, model_name: str = "gcn", num_layers: int = 2,
                                     lambda_reg: float = 0.1, epochs: int = 200,
                                     patience: int = 20, lr: float = 0.01) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = data.to(device)
    model = make_model(model_name, in_channels=data.x.size(1), num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    class_weights = _class_weights(data.y, data.train_mask).to(device)

    best_val_aucpr, best_state, stale = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        clf_loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        reg_loss = temporal_consistency_loss(out, data.edge_index, data.time_step)
        loss = clf_loss + lambda_reg * reg_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_proba = F.softmax(out[data.val_mask], dim=1)[:, 1].cpu().numpy()
            val_y = data.y[data.val_mask].cpu().numpy()
            val_aucpr = compute_metrics(val_y, (val_proba >= 0.5).astype(int), val_proba)["auc_pr"]

        if val_aucpr > best_val_aucpr:
            best_val_aucpr, best_state, stale = val_aucpr, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        test_proba = F.softmax(out[data.test_mask], dim=1)[:, 1].cpu().numpy()
        test_y = data.y[data.test_mask].cpu().numpy()
    metrics = {"model": f"{model_name}+temporal_consistency", "lambda_reg": lambda_reg,
               **compute_metrics(test_y, (test_proba >= 0.5).astype(int), test_proba)}
    return metrics


def confidence_weighted_ensemble(gnn_proba: np.ndarray, baseline_proba: np.ndarray,
                                  val_y: np.ndarray, val_gnn_proba: np.ndarray,
                                  val_baseline_proba: np.ndarray) -> tuple[np.ndarray, float]:
    """Pick the mixing weight alpha (gnn_proba*alpha + baseline_proba*(1-alpha))
    that maximizes F1 on the validation set, then apply it to the test set."""
    best_alpha, best_f1 = 0.5, -1.0
    for alpha in np.linspace(0, 1, 21):
        mixed = alpha * val_gnn_proba + (1 - alpha) * val_baseline_proba
        f1 = compute_metrics(val_y, (mixed >= 0.5).astype(int), mixed)["f1"]
        if f1 > best_f1:
            best_f1, best_alpha = f1, alpha
    test_mixed = best_alpha * gnn_proba + (1 - best_alpha) * baseline_proba
    return test_mixed, best_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", choices=["temporal", "ensemble", "both"], default="both")
    args = parser.parse_args()

    raw = load_raw()
    results = []

    if args.fix in ("temporal", "both"):
        data = build_inductive_data(raw)
        metrics = train_with_temporal_consistency(data)
        print("Temporal-consistency fix:", json.dumps(metrics, indent=2))
        results.append(metrics)
        save_metrics(metrics, RESULTS_DIR / "robustness_attempt.json")

    if args.fix in ("ensemble", "both"):
        from src.graph_data import build_pyg_data
        from src.train_gnn import train_and_eval

        data = build_inductive_data(raw)
        gnn_metrics, model = train_and_eval(data, "gcn", num_layers=2, verbose=False)

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = get_splits(raw)
        rf = train_rf(X_train, y_train)

        device = next(model.parameters()).device
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            val_gnn_proba = F.softmax(out[data.val_mask], dim=1)[:, 1].cpu().numpy()
            test_gnn_proba = F.softmax(out[data.test_mask], dim=1)[:, 1].cpu().numpy()

        val_baseline_proba = rf.predict_proba(X_val)[:, 1]
        test_baseline_proba = rf.predict_proba(X_test)[:, 1]

        test_mixed, alpha = confidence_weighted_ensemble(
            test_gnn_proba, test_baseline_proba, y_val, val_gnn_proba, val_baseline_proba)
        ensemble_metrics = {"model": "gcn+rf_ensemble", "alpha": float(alpha),
                             **compute_metrics(y_test, (test_mixed >= 0.5).astype(int), test_mixed)}
        print("Ensemble fix:", json.dumps(ensemble_metrics, indent=2))
        results.append(ensemble_metrics)
        save_metrics(ensemble_metrics, RESULTS_DIR / "robustness_attempt.json")

    md = ["# Robustness Attempt (Phase 5)", "", "Fix results next to the Phase 4 inductive baseline numbers:", ""]
    md.append("| Fix | F1 | AUC-PR |")
    md.append("|---|---|---|")
    for r in results:
        md.append(f"| {r['model']} | {r['f1']:.3f} | {r['auc_pr']:.3f} |")
    md += ["", "_Compare against results/leakage_gap_analysis.md's inductive GCN/GraphSAGE rows. "
                "Write 2-3 sentences on whether either fix closed the gap, and why or why not._"]
    (RESULTS_DIR / "robustness_attempt.md").write_text("\n".join(md) + "\n")
    print(f"Wrote {RESULTS_DIR / 'robustness_attempt.md'}")


if __name__ == "__main__":
    main()
