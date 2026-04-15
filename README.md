# Credit Risk Decision Lab

An interactive Streamlit app for credit default-risk prediction and approval decision simulation.

## Project Workflow

1. Data cleaning (`clean_data.py`)
2. Model training and comparison (`train_model.py`)
3. Interactive scoring app (`app.py`)

## Dataset

- Source file: `credit_risk_dataset.csv/credit_risk_dataset.csv`
- Source link: https://www.kaggle.com/datasets/laotse/credit-risk-dataset?resource=download
- Access date: 15 April 2026
- Target column: `loan_status` (renamed to `default_flag`)
- Main features include:
  - Applicant profile: `person_age`, `person_income`, `person_emp_length`
  - Loan profile: `loan_amnt`, `loan_int_rate`, `loan_intent`, `loan_grade`
  - Credit history: `cb_person_default_on_file`, `cb_person_cred_hist_length`

## Data Cleaning Highlights

- Standardized column names to `snake_case`
- Converted target to integer binary label (`default_flag`)
- Outlier and invalid-value handling for age, income, employment length, rate, and credit history length
- Numeric imputation with median, categorical imputation with `Unknown`
- Generated cleaning audit outputs

## Model Strategy

- Trains two models and compares performance:
  - Logistic Regression
  - Random Forest
- Excludes `loan_percent_income` during training to avoid redundant/derived feature dependence
- Uses ROC-AUC to select and save the best model
- Saves model metrics and full model comparison table

## Run Steps

```bash
pip install -r requirements.txt
python clean_data.py
python train_model.py --input data/processed/loan_clean.csv
streamlit run app.py
```

## Outputs

- `data/processed/loan_clean.csv`
- `data/processed/clean_report.csv`
- `data/processed/missing_by_column.csv`
- `models/logistic_pipeline.joblib`
- `models/random_forest_pipeline.joblib`
- `models/best_model.joblib`
- `models/model_metrics.json`
- `models/model_comparison.csv`

## Frontend Features

- Glassmorphism light theme UI (white/blue visual style)
- Model switcher in sidebar (`logistic` / `random_forest`)
- Adjustable approval threshold
- Single-applicant scoring with probability + recommendation
- Driver snapshot for quick interpretability
- Batch CSV scoring with downloadable results

## Notes

- This is an educational project for ACC102 Track 4.
- Predicted default probability is a decision-support signal, not a final lending decision.
