from pathlib import Path
import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a raw CSV file into a pandas DataFrame."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, received: {path.suffix}")

    return pd.read_csv(path)
