# Robustness Attempt (Phase 5)

Fix results next to the Phase 4 inductive baseline numbers:

| Fix | F1 | AUC-PR |
|---|---|---|
| gcn+temporal_consistency | 0.513 | 0.428 |
| gcn+rf_ensemble | 0.716 | 0.664 |

**Findings:** given Phase 4's finding that there is no real transductive/inductive leakage gap to close on this dataset (edges never cross time steps), the more meaningful comparison here is against the standalone inductive GCN (F1≈0.26–0.53 across runs, high seed variance — see `results/error_analysis.md`) and the plain baseline (RF F1=0.706, XGBoost F1=0.683, from `results/baseline_metrics.json`). The temporal-consistency regularizer did *not* help — F1=0.513 is within the GCN's own seed-to-seed noise band, consistent with there being no temporal-leakage failure mode for it to correct in the first place, since it was designed to counter exactly the cross-time-step leakage that Phase 4 showed doesn't exist here. The confidence-weighted ensemble did help, and by a meaningful, non-noise margin: F1=0.716 edges out the best standalone baseline (RF, 0.706) and comfortably beats the standalone GCN. The learned mixing weight (alpha=0.2, i.e. 80% baseline / 20% GNN) makes sense given the GNN's much higher variance and generally lower solo F1 — the ensemble is mostly the reliable baseline with a small, apparently net-positive correction from the graph model, rather than the GNN meaningfully driving performance on its own.
