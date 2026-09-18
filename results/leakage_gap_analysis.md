# Leakage-Gap Analysis (Phase 4)

**Dataset finding that reframes this analysis:** every edge in the raw Elliptic edgelist connects two nodes in the *same* time step — verified directly on the downloaded CSVs, 0 of 234,355 edges cross a time-step boundary. The transaction graph is therefore 49 disconnected components, one per time step, not one graph with edges pointing forward through time as the literature's framing assumes. A consequence: the "future-topology leakage" mechanism this phase set out to quantify (a train node's message passing reaching a test-period node via a forward-pointing edge) cannot occur here regardless of protocol, because no such edge exists. See the EDA notebook for the same check.

Comparison below uses GCN/GraphSAGE at a matched 2-layer depth (the only depth both protocols were run at) so the transductive-vs-inductive columns isolate the protocol, not architecture choice.

| Model | Transductive F1 | Inductive F1 | Shuffled-graph F1 (mean ± std) |
|---|---|---|---|
| Baseline (XGBoost) | 0.711 | 0.711 (time-based split; no graph structure involved) | n/a |
| Baseline (Random Forest) | 0.706 | 0.706 (time-based split; no graph structure involved) | n/a |
| GCN | 0.549 | 0.526 | 0.176 ± 0.018 |
| GraphSAGE | 0.292 | 0.314 | — |

## Summary

- GCN: transductive F1=0.549, inductive F1=0.526 (gap = +2.4 points, within normal seed-to-seed variance — see results/error_analysis.md for the 5-seed std).
- GraphSAGE: transductive F1=0.292, inductive F1=0.314 (gap = -2.2 points, within normal seed-to-seed variance — see results/error_analysis.md for the 5-seed std).

The degree-preserving edge-shuffle ablation (GCN, inductive protocol, 5 seeds) is the more informative test given the finding above: the real graph scores F1=0.257, versus 0.176 ± 0.018 for randomly rewired graphs with the same degree sequence. The shuffled graph underperforms the real one, which means the real transaction topology is carrying real, non-random signal under this protocol.

Net takeaway: on this dataset, the commonly-cited transductive/inductive leakage story doesn't hold at the edge level (there is no forward-pointing cross-time edge to exploit), so the real question this pipeline can actually answer is the edge-shuffle result above — whether within-time-step topology is informative at all, independent of any temporal leakage concern.
