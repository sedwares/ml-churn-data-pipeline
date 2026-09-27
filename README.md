# ML Churn Data Pipeline

A production-style machine learning data pipeline that simulates a daily CSV batch of user/patient engagement data.

## Pipeline Flow

```text
Daily raw CSV
    ↓
Ingestion
    ↓
Row-level validation
    ↓
Split records
├── valid rows   → data/processed/
└── invalid rows → data/rejected/
    ↓
Calculate batch quality percentage
    ↓
Quality gate
├── FAIL → skip training
└── PASS → transform valid rows → train churn model
```

## Data Quality Gate

The pipeline validates every row and calculates:

```text
valid percentage = valid rows / total rows × 100
```

By default, at least **90% of the batch must be valid** and at least **20 valid rows** must remain before training is allowed.

If the batch fails the quality gate, valid and rejected rows are still saved, but model training is skipped.

## Validation Rules

Rows are rejected for issues such as:

- missing `user_id`
- duplicate `user_id`
- invalid dates
- `last_active_date` earlier than `signup_date`
- negative engagement values
- non-numeric required values
- `churned` values other than `0` or `1`

Missing required columns are treated as a critical schema failure.

## Machine Learning Target

`churned` is the binary target:

- `0` = user remained active
- `1` = user churned

Only validated rows are used for feature engineering and model training.

## Synthetic Data

The CSV data in this repository is synthetic and used only to demonstrate the pipeline. No real patient or protected health information is included.

## Run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run a daily batch:

```bash
python src/pipeline.py --input data/raw/patient_engagement_2026-09-26.csv
```

Run tests:

```bash
pytest
```

## Outputs

Depending on the batch, the pipeline may create:

```text
data/processed/<input_name>_valid.csv
data/rejected/<input_name>_invalid.csv
data/processed/patient_engagement_ml_ready.csv
reports/validation_report.json
reports/training_metrics.json
models/churn_model.pkl
```
