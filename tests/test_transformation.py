import pandas as pd

from src.transformation import transform_data


def test_transformation_creates_date_features():
    df = pd.DataFrame(
        [
            {
                "user_id": 1,
                "signup_date": "2026-01-01",
                "last_active_date": "2026-09-20",
                "meals_logged_7d": 10,
                "exercise_minutes_7d": 120,
                "weight_entries_7d": 2,
                "bp_entries_7d": 3,
                "messages_sent_7d": 1,
                "support_tickets": 0,
                "churned": 0,
            }
        ]
    )

    result = transform_data(df, reference_date="2026-09-26")

    assert result.loc[0, "account_age_days"] == 268
    assert result.loc[0, "days_since_last_active"] == 6
