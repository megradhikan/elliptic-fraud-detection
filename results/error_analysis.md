# Error Analysis (Phase 6)

## Seed variance (5 seeds, GCN)

F1 = 0.358 ± 0.134, AUC-PR = 0.337 ± 0.097

## Figures

- `figures/gcn_per_time_step.png` — F1/AUC-PR per held-out time step. Highly non-stationary:
  F1 swings from ~0.52 (step 42) down to ~0.01-0.04 (steps 43-45, 47) and back up to ~0.33
  (step 48) within the same 10-step test window. Note steps 45-46 also have the thinnest
  illicit counts in the test period (5 and 2, per the EDA notebook), so those two points are
  noisy on small samples — but 43, 44, 47 all have 24+ illicit examples and still collapse,
  which is a genuine non-stationarity finding, not just a small-sample artifact. This is the
  real-world concern this project's README calls out: a single aggregate F1 hides that this
  model would have gone almost silent for several consecutive periods in this test window.
- `figures/pr_curve_comparison.png` — baseline vs. best GNN precision-recall curve. The
  baseline's curve sits above the GNN's across nearly the entire recall range, not just at
  one operating threshold — reinforcing that the F1 gap in the headline table isn't a
  threshold-tuning artifact.

## False positive / false negative inspection

| kind | node_idx | score | degree | neighbor illicit frac |
|---|---|---|---|---|
| false_positive | 144 | 0.937 | 2 | 0.00 |
| false_positive | 156 | 0.674 | 4 | 0.00 |
| false_positive | 240 | 0.791 | 22 | 0.00 |
| false_positive | 347 | 0.878 | 4 | 0.00 |
| false_positive | 456 | 0.602 | 4 | 0.00 |
| false_positive | 493 | 0.640 | 2 | 0.00 |
| false_positive | 808 | 0.604 | 2 | 0.00 |
| false_positive | 841 | 0.831 | 8 | 0.00 |
| false_positive | 902 | 0.608 | 2 | 0.00 |
| false_positive | 1072 | 0.752 | 2 | 0.00 |
| false_positive | 1299 | 0.585 | 4 | 0.00 |
| false_positive | 2058 | 0.635 | 32 | 0.00 |
| false_positive | 2083 | 0.510 | 4 | 0.00 |
| false_positive | 2182 | 0.700 | 2 | 0.00 |
| false_positive | 2193 | 0.838 | 18 | 0.00 |
| false_negative | 2613 | 0.282 | 2 | 0.00 |
| false_negative | 2718 | 0.180 | 2 | 0.00 |
| false_negative | 2719 | 0.111 | 2 | 0.00 |
| false_negative | 4545 | 0.454 | 2 | 0.00 |
| false_negative | 7297 | 0.309 | 2 | 0.00 |
| false_negative | 7299 | 0.302 | 2 | 0.00 |
| false_negative | 7300 | 0.414 | 2 | 0.00 |
| false_negative | 7302 | 0.153 | 2 | 0.00 |
| false_negative | 7303 | 0.371 | 2 | 0.00 |
| false_negative | 7304 | 0.393 | 2 | 0.00 |
| false_negative | 7306 | 0.437 | 2 | 0.00 |
| false_negative | 7308 | 0.390 | 2 | 0.00 |
| false_negative | 7309 | 0.393 | 2 | 0.00 |
| false_negative | 7315 | 0.347 | 2 | 0.00 |
| false_negative | 7321 | 0.438 | 2 | 0.00 |

**Findings:** every single FP and FN sampled above has a 0.00 neighbor-illicit-fraction — none of these errors are cases where the model got confused by a misleadingly-labeled neighborhood; it's making mistakes on nodes whose *known* local structure gave it no illicit signal at all, for better or worse. False negatives are almost all degree-2 (the minimum possible non-isolated degree), which fits the homophily finding from the EDA notebook: a node with only two neighbors, neither labeled illicit, offers the GNN essentially no structural signal to work with, so it falls back toward its raw features and under-calls the true illicit ones. False positives are more mixed — some are degree-2 like the FNs, but several (nodes 240, 2058, 2193, with degree 22/32/18) are comparatively well-connected and still misclassified, suggesting the model is not simply "confused by sparsity" in that direction; those specific high-degree FPs are worth a closer manual look (are they a specific licit transaction pattern — e.g. an exchange hot wallet — that structurally resembles illicit fan-out?) that this pipeline doesn't answer on its own.
