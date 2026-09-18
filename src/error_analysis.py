"""Phase 6: per-time-step breakdown, FP/FN inspection, PR curves, and
seed-variance — turns the headline metrics table into something you can
actually defend in an interview.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve

from src.data import TEST_TIME_STEPS, load_raw
from src.inductive_split import build_inductive_data
from src.metrics import compute_metrics
from src.train_gnn import train_and_eval

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"


def per_time_step_breakdown(data, model, model_name: str) -> None:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        proba = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    y = data.y.cpu().numpy()
    time_step = data.time_step.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    steps, f1s, aucprs = [], [], []
    for t in TEST_TIME_STEPS:
        m = test_mask & (time_step == t)
        if m.sum() == 0 or (y[m] == 1).sum() == 0:
            continue
        preds = (proba[m] >= 0.5).astype(int)
        metrics = compute_metrics(y[m], preds, proba[m])
        steps.append(t)
        f1s.append(metrics["f1"])
        aucprs.append(metrics["auc_pr"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, f1s, marker="o", label="F1")
    ax.plot(steps, aucprs, marker="s", label="AUC-PR")
    ax.set_xlabel("Time step (test period)")
    ax.set_ylabel("Score")
    ax.set_title(f"{model_name}: per-time-step performance on illicit class")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{model_name}_per_time_step.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / f'{model_name}_per_time_step.png'}")


def pr_curve_comparison(y_test, baseline_proba, gnn_proba) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba in (("Baseline (best)", baseline_proba), ("Best GNN", gnn_proba)):
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=name)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — illicit class")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "pr_curve_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'pr_curve_comparison.png'}")


def seed_variance(data, model_name: str, seeds=(0, 1, 2, 3, 4)) -> dict:
    f1s, aucprs = [], []
    for seed in seeds:
        torch.manual_seed(seed)
        metrics, _ = train_and_eval(data, model_name, num_layers=2, verbose=False)
        f1s.append(metrics["f1"])
        aucprs.append(metrics["auc_pr"])
    return {
        "model": model_name, "seeds": list(seeds),
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "auc_pr_mean": float(np.mean(aucprs)), "auc_pr_std": float(np.std(aucprs)),
    }


def fp_fn_inspection(data, model, k: int = 15) -> list[dict]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        proba = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    y = data.y.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()
    preds = (proba >= 0.5).astype(int)

    fp_idx = np.where(test_mask & (preds == 1) & (y == 0))[0]
    fn_idx = np.where(test_mask & (preds == 0) & (y == 1))[0]

    src, dst = data.edge_index[0].numpy(), data.edge_index[1].numpy()
    degree = np.bincount(np.concatenate([src, dst]), minlength=data.num_nodes)

    def describe(idx_arr, kind):
        rows = []
        for i in idx_arr[:k]:
            neighbors = np.concatenate([dst[src == i], src[dst == i]])
            neighbor_labels = y[neighbors]
            illicit_frac = float((neighbor_labels == 1).mean()) if len(neighbor_labels) else None
            rows.append({
                "kind": kind, "node_idx": int(i), "score": float(proba[i]),
                "degree": int(degree[i]), "neighbor_illicit_frac": illicit_frac,
            })
        return rows

    return describe(fp_idx, "false_positive") + describe(fn_idx, "false_negative")


def main():
    raw = load_raw()
    data = build_inductive_data(raw)

    print("Training best GNN (GCN, 2 layers) for error analysis...")
    gnn_metrics, model = train_and_eval(data, "gcn", num_layers=2)
    per_time_step_breakdown(data, model, "gcn")

    from src.baseline import get_splits, train_rf
    (X_train, y_train), _, (X_test, y_test) = get_splits(raw)
    rf = train_rf(X_train, y_train)
    baseline_proba = rf.predict_proba(X_test)[:, 1]

    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        gnn_test_proba = F.softmax(out[data.test_mask], dim=1)[:, 1].cpu().numpy()
    pr_curve_comparison(y_test, baseline_proba, gnn_test_proba)

    variance = seed_variance(data, "gcn")
    print("Seed variance:", json.dumps(variance, indent=2))

    fp_fn = fp_fn_inspection(data, model)

    report = ["# Error Analysis (Phase 6)", "",
              "## Seed variance (5 seeds, GCN)", "",
              f"F1 = {variance['f1_mean']:.3f} ± {variance['f1_std']:.3f}, "
              f"AUC-PR = {variance['auc_pr_mean']:.3f} ± {variance['auc_pr_std']:.3f}", "",
              "## Figures", "",
              "- `figures/gcn_per_time_step.png` — F1/AUC-PR per held-out time step",
              "- `figures/pr_curve_comparison.png` — baseline vs. best GNN precision-recall curve",
              "", "## False positive / false negative inspection", "",
              "| kind | node_idx | score | degree | neighbor illicit frac |",
              "|---|---|---|---|---|"]
    for row in fp_fn:
        frac = f"{row['neighbor_illicit_frac']:.2f}" if row["neighbor_illicit_frac"] is not None else "n/a"
        report.append(f"| {row['kind']} | {row['node_idx']} | {row['score']:.3f} | {row['degree']} | {frac} |")
    report += ["", "_Write 3-5 sentences here on any pattern you see in the FP/FN table above "
                    "(e.g. do FNs cluster at low degree / low neighbor-illicit-fraction, suggesting "
                    "the model under-uses weak structural signal for borderline cases?)._"]

    (RESULTS_DIR / "error_analysis.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {RESULTS_DIR / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
