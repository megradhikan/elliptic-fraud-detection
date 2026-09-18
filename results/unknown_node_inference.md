# Unknown-Node Inference (Phase 7)

Scored 157205 unlabeled nodes (mean=0.413, median=0.391, p95=0.916).

See `figures/unknown_node_score_distribution.png` for the full distribution. **It is not the plausible mostly-low-risk shape hoped for**: there's a large spike near 0 (~21K nodes), but then a long, roughly flat plateau across the entire 0.1–1.0 range rather than a small, clearly-separated high-risk tail. That's a real limitation worth stating plainly: the inductive GCN (already shown to have high seed-to-seed variance in `results/error_analysis.md`, F1 std=0.134) does not appear well-calibrated on the ~77% of nodes it never saw a label for during training — it's confidently placing a large fraction of ordinary unlabeled transactions in the middle-to-high risk range, which is not something you'd want to threshold on for a production alerting queue without recalibration (e.g. Platt scaling / isotonic regression on a held-out labeled set) first.

## Top 10 highest-scored unlabeled nodes

| node_idx | score | degree | neighbors | illicit frac (of known neighbors) |
|---|---|---|---|---|
| 42577 | 1.000 | 2 | 2 | n/a |
| 42581 | 1.000 | 2 | 2 | n/a |
| 189711 | 1.000 | 116 | 116 | 1.00 |
| 42693 | 1.000 | 2 | 2 | n/a |
| 42573 | 1.000 | 2 | 2 | n/a |
| 42576 | 1.000 | 2 | 2 | n/a |
| 42692 | 1.000 | 2 | 2 | n/a |
| 42691 | 1.000 | 2 | 2 | n/a |
| 42690 | 1.000 | 2 | 2 | n/a |
| 42689 | 1.000 | 2 | 2 | n/a |

**Findings:** the top-10 list splits into two distinct patterns. Node `189711` is the standout: degree 116, and 100% of its 116 neighbors that do have known labels are illicit — exactly the kind of dense, illicit-surrounded structure Phase 6's true-positive cases showed, and a strong, structurally-grounded candidate for manual review. The other nine nodes all have degree 2, no labeled neighbors at all (`n/a`), and every one scored *exactly* 1.000 — that uniform, saturated score with zero structural signal to justify it looks like a model-confidence artifact rather than a genuine risk signal (consistent with the calibration concern above), not a pattern worth acting on without the recalibration step already flagged.
