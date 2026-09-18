"""Phase 4 exit criteria: assemble the transductive vs. inductive vs.
shuffled-graph comparison table into results/leakage_gap_analysis.md.

Run after baseline.py, train_gnn.py (transductive), inductive_split.py, and
edge_shuffle.py have all written their metrics files.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _best_by_f1(rows: list[dict], **filters) -> dict | None:
    candidates = [r for r in rows if all(r.get(k) == v for k, v in filters.items())]
    return max(candidates, key=lambda r: r["f1"], default=None)


def main():
    baseline = _load(RESULTS_DIR / "baseline_metrics.json")
    gnn = _load(RESULTS_DIR / "gnn_metrics.json")
    shuffle = _load(RESULTS_DIR / "edge_shuffle_ablation.json")

    rf_row = _best_by_f1(baseline, model="random_forest")
    xgb_row = _best_by_f1(baseline, model="xgboost")

    lines = ["# Leakage-Gap Analysis (Phase 4)", ""]
    lines.append("| Model | Transductive F1 | Inductive F1 | Shuffled-graph F1 (mean ± std) |")
    lines.append("|---|---|---|---|")

    if xgb_row:
        lines.append(f"| Baseline (XGBoost) | {xgb_row['f1']:.3f} | {xgb_row['f1']:.3f} (leakage-free by construction) | n/a |")
    if rf_row:
        lines.append(f"| Baseline (Random Forest) | {rf_row['f1']:.3f} | {rf_row['f1']:.3f} (leakage-free by construction) | n/a |")

    for model_name in ("gcn", "graphsage"):
        trans = _best_by_f1(gnn, model=model_name, protocol="transductive")
        induct = _best_by_f1(gnn, model=model_name, protocol="inductive")
        shuf = next((s for s in shuffle if s.get("model") == model_name), None)
        trans_f1 = f"{trans['f1']:.3f}" if trans else "—"
        induct_f1 = f"{induct['f1']:.3f}" if induct else "—"
        shuf_f1 = f"{shuf['shuffled_graph']['f1_mean']:.3f} ± {shuf['shuffled_graph']['f1_std']:.3f}" if shuf else "—"
        label = "GCN" if model_name == "gcn" else "GraphSAGE"
        lines.append(f"| {label} | {trans_f1} | {induct_f1} | {shuf_f1} |")

    gap_lines = []
    for model_name, label in (("gcn", "GCN"), ("graphsage", "GraphSAGE")):
        trans = _best_by_f1(gnn, model=model_name, protocol="transductive")
        induct = _best_by_f1(gnn, model=model_name, protocol="inductive")
        if trans and induct:
            gap = (trans["f1"] - induct["f1"]) * 100
            gap_lines.append(f"- {label}: transductive-to-inductive F1 gap = {gap:.1f} points")

    lines += ["", "## Summary", ""] + gap_lines + [
        "",
        "_Fill in 2-3 sentences here once the runs complete: does the leakage gap "
        "replicate on this pipeline, and does the shuffled graph match or beat the "
        "real graph under the inductive protocol (per Maganti 2026)?_",
    ]

    out_path = RESULTS_DIR / "leakage_gap_analysis.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
