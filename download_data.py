"""Download the Elliptic Data Set from Kaggle into data/raw/.

Kaggle requires an authenticated account (the dataset's license/terms must be
accepted via the Kaggle website at least once), so this cannot be fully
automated without your credentials.

Setup (one-time):
  1. Create a Kaggle account and go to https://www.kaggle.com/settings/account
  2. Under "API", click "Create New Token" -> downloads kaggle.json
  3. Move it to ~/.kaggle/kaggle.json and `chmod 600 ~/.kaggle/kaggle.json`
     (or set KAGGLE_USERNAME / KAGGLE_KEY env vars instead)
  4. Visit https://www.kaggle.com/datasets/ellipticco/elliptic-data-set once
     in a browser and accept the dataset's terms if prompted.

Then run:
    pip install kaggle
    python download_data.py

This writes elliptic_txs_features.csv, elliptic_txs_edgelist.csv, and
elliptic_txs_classes.csv into data/raw/.

If you'd rather not use the API, download the dataset zip manually from the
URL above, unzip it, and place the three CSVs directly in data/raw/ — the
rest of the pipeline only cares that those files exist there.
"""

import shutil
import sys
import zipfile
from pathlib import Path

DATASET = "ellipticco/elliptic-data-set"
RAW_DIR = Path(__file__).parent / "data" / "raw"
EXPECTED_FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if all((RAW_DIR / f).exists() for f in EXPECTED_FILES):
        print(f"All expected files already present in {RAW_DIR}, nothing to do.")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except (ImportError, OSError) as e:
        sys.exit(
            "Could not import/authenticate the kaggle package "
            f"({e}).\nInstall it with `pip install kaggle` and set up "
            "~/.kaggle/kaggle.json as described in this file's docstring, "
            "then re-run. Alternatively, download the dataset manually from "
            f"https://www.kaggle.com/datasets/{DATASET} and place the CSVs "
            f"in {RAW_DIR}."
        )

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET} ...")
    api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=False, quiet=False)

    zip_path = RAW_DIR / "elliptic-data-set.zip"
    if not zip_path.exists():
        zips = list(RAW_DIR.glob("*.zip"))
        if not zips:
            sys.exit(f"Expected a downloaded zip in {RAW_DIR}, found none.")
        zip_path = zips[0]

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()

    # Kaggle sometimes nests files under an extra directory (e.g. elliptic_bitcoin_dataset/) — flatten.
    for f in EXPECTED_FILES:
        if not (RAW_DIR / f).exists():
            found = list(RAW_DIR.rglob(f))
            if found:
                shutil.move(str(found[0]), str(RAW_DIR / f))

    missing = [f for f in EXPECTED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        sys.exit(f"Download finished but still missing: {missing}. Check {RAW_DIR} manually.")

    print(f"Done. Files present in {RAW_DIR}:")
    for f in EXPECTED_FILES:
        print(f"  {f}")


if __name__ == "__main__":
    main()
