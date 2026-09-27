import pandas as pd

from src.validation import validate_rows


def make_mixed_df() -> pd.DataFrame:
    rows = []

    for i in range(18):
        rows.append({
            "user_id": 1000 + i,
            "signup_date": "2026-01-01",
            "last_active_date": "2026-09-20",
            "meals_logged_7d": 10,
            "exercise_minutes_7d": 120,
            "weight_entries_7d": 2,
            "bp_entries_7d": 3,
            "messages_sent_7d": 1,
            "support_tickets": 0,
            "churned": i % 2,
        })

    rows.append({
        "user_id": 2001,
        "signup_date": "2026-01-01",
        "last_active_date": "2026-09-20",
        "meals_logged_7d": 10,
        "exercise_minutes_7d": -10,
        "weight_entries_7d": 2,
        "bp_entries_7d": 3,
        "messages_sent_7d": 1,
        "support_tickets": 0,
        "churned": 1,
    })

    rows.append({
        "user_id": 2002,
        "signup_date": "bad-date",
        "last_active_date": "2026-09-20",
        "meals_logged_7d": 10,
        "exercise_minutes_7d": 120,
        "weight_entries_7d": 2,
        "bp_entries_7d": 3,
        "messages_sent_7d": 1,
        "support_tickets": 0,
        "churned": "yes",
    })

    return pd.DataFrame(rows)


def test_rows_are_split_into_valid_and_invalid():
    result = validate_rows(make_mixed_df())
    assert result.total_rows == 20
    assert result.valid_count == 18
    assert result.invalid_count == 2
    assert result.valid_percentage == 90.0


def test_invalid_rows_include_reasons():
    result = validate_rows(make_mixed_df())
    reasons = " | ".join(result.invalid_rows["validation_errors"].tolist())
    assert "exercise_minutes_7d is negative" in reasons
    assert "invalid signup_date" in reasons
    assert "churned must be 0 or 1" in reasons


def test_missing_required_column_is_schema_failure():
    df = make_mixed_df().drop(columns=["support_tickets"])
    result = validate_rows(df)
    assert result.valid_count == 0
    assert result.schema_errors
    assert "Missing required columns" in result.schema_errors[0]
