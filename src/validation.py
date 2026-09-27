from dataclasses import dataclass
import pandas as pd


REQUIRED_COLUMNS = [
    "user_id",
    "signup_date",
    "last_active_date",
    "meals_logged_7d",
    "exercise_minutes_7d",
    "weight_entries_7d",
    "bp_entries_7d",
    "messages_sent_7d",
    "support_tickets",
    "churned",
]

NON_NEGATIVE_COLUMNS = [
    "meals_logged_7d",
    "exercise_minutes_7d",
    "weight_entries_7d",
    "bp_entries_7d",
    "messages_sent_7d",
    "support_tickets",
]


@dataclass
class ValidationResult:
    valid_rows: pd.DataFrame
    invalid_rows: pd.DataFrame
    total_rows: int
    valid_count: int
    invalid_count: int
    valid_percentage: float
    schema_errors: list[str]


def _row_errors(row: pd.Series) -> list[str]:
    """Return all validation errors for a single row."""
    errors: list[str] = []

    if pd.isna(row["user_id"]):
        errors.append("missing user_id")

    signup_date = pd.to_datetime(row["signup_date"], errors="coerce")
    last_active_date = pd.to_datetime(row["last_active_date"], errors="coerce")

    if pd.isna(signup_date):
        errors.append("invalid signup_date")

    if pd.isna(last_active_date):
        errors.append("invalid last_active_date")

    if pd.notna(signup_date) and pd.notna(last_active_date):
        if last_active_date < signup_date:
            errors.append("last_active_date before signup_date")

    for column in NON_NEGATIVE_COLUMNS:
        value = pd.to_numeric(row[column], errors="coerce")
        if pd.isna(value):
            errors.append(f"{column} is missing or non-numeric")
        elif value < 0:
            errors.append(f"{column} is negative")

    churn_value = pd.to_numeric(row["churned"], errors="coerce")
    if pd.isna(churn_value) or churn_value not in [0, 1]:
        errors.append("churned must be 0 or 1")

    return errors


def validate_rows(df: pd.DataFrame) -> ValidationResult:
    """Validate each row, then split the batch into valid and invalid records."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        message = f"Missing required columns: {missing_columns}"
        invalid_rows = df.copy()
        invalid_rows["validation_errors"] = message
        return ValidationResult(
            valid_rows=df.iloc[0:0].copy(),
            invalid_rows=invalid_rows,
            total_rows=len(df),
            valid_count=0,
            invalid_count=len(df),
            valid_percentage=0.0,
            schema_errors=[message],
        )

    working = df.copy()
    duplicate_mask = working["user_id"].duplicated(keep=False)
    row_error_lists: list[list[str]] = []

    for index, row in working.iterrows():
        errors = _row_errors(row)
        if duplicate_mask.loc[index] and pd.notna(row["user_id"]):
            errors.append("duplicate user_id")
        row_error_lists.append(errors)

    working["validation_errors"] = [
        "; ".join(errors) if errors else "" for errors in row_error_lists
    ]

    invalid_mask = working["validation_errors"] != ""
    invalid_rows = working.loc[invalid_mask].copy()
    valid_rows = working.loc[~invalid_mask].drop(columns=["validation_errors"]).copy()

    total_rows = len(working)
    valid_count = len(valid_rows)
    invalid_count = len(invalid_rows)
    valid_percentage = round((valid_count / total_rows * 100), 2) if total_rows else 0.0

    return ValidationResult(
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        total_rows=total_rows,
        valid_count=valid_count,
        invalid_count=invalid_count,
        valid_percentage=valid_percentage,
        schema_errors=[],
    )
