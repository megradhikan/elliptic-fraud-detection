"""Load and merge the raw Elliptic CSVs into a single tabular object.

Raw label convention in elliptic_txs_classes.csv: '1' = illicit, '2' = licit,
'unknown' = unlabeled. We remap to {1: illicit, 0: licit, -1: unknown} so the
illicit class is always the positive class for sklearn/xgboost metrics.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}
ILLICIT, LICIT, UNKNOWN = 1, 0, -1

# Time-based split (Phase 1, task 5): train on early time steps, validate and
# test on later, disjoint ones so no future information leaks into training.
# The Elliptic dataset has 49 time steps, each with a reasonably stable ~2%
# illicit rate, so an even 34/5/10 split keeps a usable number of illicit
# examples in every split.
TRAIN_TIME_STEPS = list(range(1, 35))
VAL_TIME_STEPS = list(range(35, 40))
TEST_TIME_STEPS = list(range(40, 50))


@dataclass
class EllipticData:
    """Node features/labels/time steps plus the transaction edge list.

    features: DataFrame indexed by txId, columns = 166 feature columns
              (time_step is included as its own column, not counted in the 166)
    labels: Series indexed by txId, values in {1 illicit, 0 licit, -1 unknown}
    edges: DataFrame with columns [src, dst], both txIds, directed src->dst
    """

    features: pd.DataFrame
    labels: pd.Series
    edges: pd.DataFrame

    @property
    def feature_cols(self) -> list[str]:
        return [c for c in self.features.columns if c != "time_step"]

    def node_ids(self) -> np.ndarray:
        return self.features.index.to_numpy()

    def split_ids(self, time_steps: list[int]) -> np.ndarray:
        mask = self.features["time_step"].isin(time_steps)
        return self.features.index[mask].to_numpy()

    def labeled_mask(self) -> pd.Series:
        return self.labels != UNKNOWN


def load_raw(raw_dir: Path = RAW_DIR) -> EllipticData:
    features_path = raw_dir / "elliptic_txs_features.csv"
    edges_path = raw_dir / "elliptic_txs_edgelist.csv"
    classes_path = raw_dir / "elliptic_txs_classes.csv"

    for p in (features_path, edges_path, classes_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `python download_data.py` first "
                "(see that file's docstring for Kaggle auth setup)."
            )

    # The features CSV ships with no header: col 0 = txId, col 1 = time step,
    # remaining columns = local + one-hop-aggregated features (~166, exact
    # count inferred from the file rather than hardcoded to tolerate minor
    # version differences in the Kaggle mirror).
    raw = pd.read_csv(features_path, header=None)
    n_features = raw.shape[1] - 2
    raw.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, n_features + 1)]
    features = raw.set_index("txId").sort_index()

    edges = pd.read_csv(edges_path)
    edges.columns = ["src", "dst"]

    classes = pd.read_csv(classes_path)
    classes.columns = ["txId", "class"]
    classes = classes.set_index("txId")
    labels = classes["class"].astype(str).map(LABEL_MAP)
    labels = labels.reindex(features.index)

    return EllipticData(features=features, labels=labels, edges=edges)


if __name__ == "__main__":
    data = load_raw()
    print(f"nodes={len(data.features)} edges={len(data.edges)} "
          f"illicit={(data.labels == ILLICIT).sum()} "
          f"licit={(data.labels == LICIT).sum()} "
          f"unknown={(data.labels == UNKNOWN).sum()}")
