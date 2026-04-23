"""
loader.py — Download or validate the raw Telco Churn dataset.

Run standalone:  python -m src.data.loader
"""

import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import RAW_FILE, DATA_RAW

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def download_dataset(url: str = DATASET_URL, dest: Path = RAW_FILE) -> Path:
    """Download raw CSV if it doesn't already exist locally."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[loader] Dataset already present at {dest}")
        return dest
    print(f"[loader] Downloading dataset from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[loader] Saved to {dest}")
    return dest


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Return the raw DataFrame, downloading it first if needed."""
    if not path.exists():
        download_dataset(dest=path)
    df = pd.read_csv(path)
    print(f"[loader] Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def validate(df: pd.DataFrame) -> None:
    """Raise if critical columns are missing."""
    required = {"customerID", "Churn", "MonthlyCharges", "TotalCharges", "tenure"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[loader] Missing expected columns: {missing}")
    print("[loader] Schema validation passed.")


if __name__ == "__main__":
    raw = load_raw()
    validate(raw)
    print(raw.head(3))
