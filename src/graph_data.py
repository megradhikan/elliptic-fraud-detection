"""Build a torch_geometric.data.Data object from an EllipticData.

Node ordering is fixed by sorting txIds once here; every mask/index derived
from EllipticData elsewhere must use this same ordering to stay aligned.
"""

import numpy as np
import torch
from torch_geometric.data import Data

from src.data import EllipticData, TEST_TIME_STEPS, TRAIN_TIME_STEPS, VAL_TIME_STEPS


def build_pyg_data(data: EllipticData) -> Data:
    node_ids = data.node_ids()
    id_to_idx = {tx_id: i for i, tx_id in enumerate(node_ids)}

    x = torch.tensor(data.features[data.feature_cols].to_numpy(), dtype=torch.float)
    y = torch.tensor(data.labels.to_numpy(), dtype=torch.long)
    time_step = torch.tensor(data.features["time_step"].to_numpy(), dtype=torch.long)

    # Drop edges whose endpoints aren't in the feature table (shouldn't
    # normally happen, but the raw edgelist isn't guaranteed to be a strict
    # subset in every Kaggle mirror).
    valid = data.edges["src"].isin(id_to_idx) & data.edges["dst"].isin(id_to_idx)
    edges = data.edges[valid]
    src = edges["src"].map(id_to_idx).to_numpy()
    dst = edges["dst"].map(id_to_idx).to_numpy()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    labeled = y != -1
    train_mask = _mask_for(time_step, TRAIN_TIME_STEPS) & labeled
    val_mask = _mask_for(time_step, VAL_TIME_STEPS) & labeled
    test_mask = _mask_for(time_step, TEST_TIME_STEPS) & labeled

    pyg_data = Data(x=x, edge_index=edge_index, y=y)
    pyg_data.time_step = time_step
    pyg_data.train_mask = train_mask
    pyg_data.val_mask = val_mask
    pyg_data.test_mask = test_mask
    pyg_data.node_ids = torch.tensor(node_ids, dtype=torch.long)
    return pyg_data


def _mask_for(time_step: torch.Tensor, steps: list[int]) -> torch.Tensor:
    steps_t = torch.tensor(steps, dtype=time_step.dtype)
    return torch.isin(time_step, steps_t)


if __name__ == "__main__":
    from src.data import load_raw

    raw = load_raw()
    g = build_pyg_data(raw)
    print(g)
    print(f"train={int(g.train_mask.sum())} val={int(g.val_mask.sum())} test={int(g.test_mask.sum())}")
