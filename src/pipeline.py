from pathlib import Path
import argparse
import yaml

from ingestion import load_csv
from validation import validate_rows
from transformation import transform_data
from storage import save_dataframe, save_json
from training import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open() as file:
        return yaml.safe_load(file)


def run_pipeline(input_file: str) -> None:
    config = load_config()
    input_path = PROJECT_ROOT / input_file

    df = load_csv(input_path)
    print(f"[INGESTION] Loaded {len(df)} rows from {input_path.name}")

    validation = validate_rows(df)

    valid_output_path = (
        PROJECT_ROOT / "data" / "processed" / f"{input_path.stem}_valid.csv"
    )
    invalid_output_path = (
        PROJECT_ROOT / "data" / "rejected" / f"{input_path.stem}_invalid.csv"
    )

    save_dataframe(validation.valid_rows, valid_output_path)
    if validation.invalid_count > 0:
        save_dataframe(validation.invalid_rows, invalid_output_path)

    minimum_quality = config["validation"]["min_valid_percentage"]
    minimum_rows = config["validation"]["min_valid_rows_for_training"]

    report = {
        "input_file": input_path.name,
        "total_rows": validation.total_rows,
        "valid_rows": validation.valid_count,
        "invalid_rows": validation.invalid_count,
        "valid_percentage": validation.valid_percentage,
        "minimum_required_percentage": minimum_quality,
        "minimum_valid_rows_for_training": minimum_rows,
        "schema_errors": validation.schema_errors,
    }
    save_json(report, PROJECT_ROOT / config["paths"]["validation_report"])

    print(
        f"[VALIDATION] {validation.valid_count}/{validation.total_rows} rows valid "
        f"({validation.valid_percentage}%)"
    )
    print(f"[STORAGE] Valid rows saved to {valid_output_path}")
    if validation.invalid_count > 0:
        print(f"[STORAGE] Invalid rows saved to {invalid_output_path}")

    if validation.schema_errors:
        print("[QUALITY GATE] FAILED because required columns are missing.")
        print("[TRAINING] Skipped.")
        return

    if validation.valid_percentage < minimum_quality:
        print(
            f"[QUALITY GATE] FAILED: {validation.valid_percentage}% valid "
            f"< required {minimum_quality}%"
        )
        print("[TRAINING] Skipped because batch quality is below threshold.")
        return

    if validation.valid_count < minimum_rows:
        print(
            f"[QUALITY GATE] FAILED: only {validation.valid_count} valid rows; "
            f"{minimum_rows} required."
        )
        print("[TRAINING] Skipped because too few valid rows remain.")
        return

    print("[QUALITY GATE] PASSED")

    transformed_df = transform_data(
        validation.valid_rows,
        reference_date=config["pipeline"]["reference_date"],
    )

    ml_ready_path = PROJECT_ROOT / config["paths"]["ml_ready_data"]
    save_dataframe(transformed_df, ml_ready_path)
    print(f"[TRANSFORMATION] ML-ready data saved to {ml_ready_path}")

    model_path = PROJECT_ROOT / config["paths"]["model"]
    metrics = train_model(transformed_df, model_path=model_path)

    metrics_path = PROJECT_ROOT / config["paths"]["training_metrics"]
    save_json(metrics, metrics_path)

    print(f"[TRAINING] Model saved to {model_path}")
    print(f"[TRAINING] Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn ML data pipeline.")
    parser.add_argument(
        "--input",
        default="data/raw/patient_engagement_2026-09-26.csv",
        help="Input CSV path relative to the project root.",
    )
    args = parser.parse_args()
    run_pipeline(args.input)
