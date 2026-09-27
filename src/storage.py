from pathlib import Path
import json
import pandas as pd


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a DataFrame as CSV, creating parent directories if necessary."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_json(data: dict, output_path: str | Path) -> Path:
    """Save a dictionary as a formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as file:
        json.dump(data, file, indent=2)

    return path
