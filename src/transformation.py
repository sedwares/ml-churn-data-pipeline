import pandas as pd


MODEL_FEATURES = [
    "meals_logged_7d",
    "exercise_minutes_7d",
    "weight_entries_7d",
    "bp_entries_7d",
    "messages_sent_7d",
    "support_tickets",
    "account_age_days",
    "days_since_last_active",
]

TARGET_COLUMN = "churned"

NUMERIC_SOURCE_COLUMNS = [
    "meals_logged_7d",
    "exercise_minutes_7d",
    "weight_entries_7d",
    "bp_entries_7d",
    "messages_sent_7d",
    "support_tickets",
    TARGET_COLUMN,
]


def transform_data(df: pd.DataFrame, reference_date: str = "2026-09-26") -> pd.DataFrame:
    """Convert validated raw rows into ML-ready numeric features."""
    transformed = df.copy()

    transformed["signup_date"] = pd.to_datetime(transformed["signup_date"])
    transformed["last_active_date"] = pd.to_datetime(transformed["last_active_date"])

    for column in NUMERIC_SOURCE_COLUMNS:
        transformed[column] = pd.to_numeric(transformed[column], errors="raise")

    # Keep the binary target explicitly integer-typed for classification metrics.
    transformed[TARGET_COLUMN] = transformed[TARGET_COLUMN].astype(int)

    reference = pd.Timestamp(reference_date)

    transformed["account_age_days"] = (
        reference - transformed["signup_date"]
    ).dt.days

    transformed["days_since_last_active"] = (
        reference - transformed["last_active_date"]
    ).dt.days

    model_columns = ["user_id"] + MODEL_FEATURES + [TARGET_COLUMN]
    return transformed[model_columns]
