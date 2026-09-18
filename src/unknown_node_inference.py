"""Phase 7: score the ~77% unlabeled nodes — the actual point of a model like
this in production, since you never have labels for genuinely new activity.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import UNKNOWN, load_raw
from src.inductive_split import build_inductive_data
from src.train_gnn import train_and_eval

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"


def score_unknown_nodes(data, model) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        proba = F.softmax(out, dim=1)[:, 1].cpu().numpy()
    unknown_mask = (data.y == UNKNOWN).numpy()
    unknown_idx = np.where(unknown_mask)[0]
    return unknown_idx, proba[unknown_idx]


def plot_score_distribution(scores: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=50)
    ax.set_xlabel("Predicted illicit probability")
    ax.set_ylabel("Count (unlabeled nodes)")
    ax.set_title("Risk score distribution over unlabeled nodes")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "unknown_node_score_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIG_DIR / 'unknown_node_score_distribution.png'}")


def inspect_top_scored(data, unknown_idx: np.ndarray, scores: np.ndarray, k: int = 10) -> list[dict]:
    order = np.argsort(scores)[::-1][:k]
    top_nodes = unknown_idx[order]
    top_scores = scores[order]

    src, dst = data.edge_index[0].numpy(), data.edge_index[1].numpy()
    y = data.y.numpy()
    degree = np.bincount(np.concatenate([src, dst]), minlength=data.num_nodes)

    rows = []
    for node, score in zip(top_nodes, top_scores):
        neighbors = np.concatenate([dst[src == node], src[dst == node]])
        neighbor_labels = y[neighbors]
        known = neighbor_labels[neighbor_labels != UNKNOWN]
        illicit_frac = float((known == 1).mean()) if len(known) else None
        rows.append({
            "node_idx": int(node), "score": float(score), "degree": int(degree[node]),
            "n_neighbors": int(len(neighbors)), "neighbor_illicit_frac_known": illicit_frac,
        })
    return rows


def main():
    raw = load_raw()
    data = build_inductive_data(raw)

    print("Training GCN for unknown-node inference...")
    _, model = train_and_eval(data, "gcn", num_layers=2)

    unknown_idx, scores = score_unknown_nodes(data, model)
    print(f"Scored {len(unknown_idx)} unlabeled nodes. "
          f"mean={scores.mean():.3f} median={np.median(scores):.3f} "
          f"p95={np.percentile(scores, 95):.3f}")
    plot_score_distribution(scores)

    top = inspect_top_scored(data, unknown_idx, scores)

    report = ["# Unknown-Node Inference (Phase 7)", "",
              f"Scored {len(unknown_idx)} unlabeled nodes "
              f"(mean={scores.mean():.3f}, median={np.median(scores):.3f}, "
              f"p95={np.percentile(scores, 95):.3f}).", "",
              "See `figures/unknown_node_score_distribution.png` for the full distribution — "
              "check it looks like a plausible mostly-low-risk mix with a small high-risk tail, "
              "not a degenerate spike near 0.5.", "",
              "## Top 10 highest-scored unlabeled nodes", "",
              "| node_idx | score | degree | neighbors | illicit frac (of known neighbors) |",
              "|---|---|---|---|---|"]
    for row in top:
        frac = f"{row['neighbor_illicit_frac_known']:.2f}" if row["neighbor_illicit_frac_known"] is not None else "n/a"
        report.append(f"| {row['node_idx']} | {row['score']:.3f} | {row['degree']} | "
                       f"{row['n_neighbors']} | {frac} |")
    report += ["", "_Write down whether the top-scored nodes' local structure (fan-out, degree, "
                    "neighbor risk) looks similar to the known-illicit nodes inspected in Phase 6's "
                    "FP/FN analysis._"]

    (RESULTS_DIR / "unknown_node_inference.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {RESULTS_DIR / 'unknown_node_inference.md'}")


if __name__ == "__main__":
    main()
