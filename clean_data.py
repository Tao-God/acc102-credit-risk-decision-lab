import argparse
import os

import numpy as np
import pandas as pd


def to_snake(cols):
    return [c.strip().lower().replace(" ", "_").replace("-", "_") for c in cols]


def clean_credit_risk_data(df: pd.DataFrame):
    report_rows = []

    df = df.copy()
    report_rows.append({"step": "raw_shape", "value": f"{len(df)}x{len(df.columns)}"})
    df.columns = to_snake(df.columns)

    if "loan_status" not in df.columns:
        raise ValueError("Column 'loan_status' not found.")

    df = df.rename(columns={"loan_status": "default_flag"})

    before_dup = len(df)
    df = df.drop_duplicates()
    report_rows.append({"step": "drop_duplicates", "value": int(before_dup - len(df))})

    if "id" in df.columns:
        df = df.drop(columns=["id"])
        report_rows.append({"step": "drop_id", "value": 1})

    df["default_flag"] = pd.to_numeric(df["default_flag"], errors="coerce")
    bad_target = int(df["default_flag"].isna().sum())
    df = df.dropna(subset=["default_flag"])
    df["default_flag"] = df["default_flag"].astype(int)
    report_rows.append({"step": "drop_target_na", "value": bad_target})

    numeric_cols = [
        "person_age",
        "person_income",
        "person_emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    clean_rules = [
        ("person_age", df["person_age"] < 18 if "person_age" in df.columns else pd.Series(dtype=bool), "age_lt_18_to_nan"),
        ("person_age", df["person_age"] > 100 if "person_age" in df.columns else pd.Series(dtype=bool), "age_gt_100_to_nan"),
        ("person_income", df["person_income"] <= 0 if "person_income" in df.columns else pd.Series(dtype=bool), "income_non_positive_to_nan"),
        ("person_emp_length", df["person_emp_length"] < 0 if "person_emp_length" in df.columns else pd.Series(dtype=bool), "emp_length_negative_to_nan"),
        ("person_emp_length", df["person_emp_length"] > 60 if "person_emp_length" in df.columns else pd.Series(dtype=bool), "emp_length_gt_60_to_nan"),
        ("loan_amnt", df["loan_amnt"] <= 0 if "loan_amnt" in df.columns else pd.Series(dtype=bool), "loan_amnt_non_positive_to_nan"),
        ("loan_int_rate", (df["loan_int_rate"] <= 0) | (df["loan_int_rate"] > 60) if "loan_int_rate" in df.columns else pd.Series(dtype=bool), "loan_int_rate_out_of_range_to_nan"),
        ("loan_percent_income", (df["loan_percent_income"] <= 0) | (df["loan_percent_income"] > 1.2) if "loan_percent_income" in df.columns else pd.Series(dtype=bool), "loan_percent_income_out_of_range_to_nan"),
        ("cb_person_cred_hist_length", df["cb_person_cred_hist_length"] < 0 if "cb_person_cred_hist_length" in df.columns else pd.Series(dtype=bool), "cred_hist_negative_to_nan"),
        ("cb_person_cred_hist_length", df["cb_person_cred_hist_length"] > 80 if "cb_person_cred_hist_length" in df.columns else pd.Series(dtype=bool), "cred_hist_gt_80_to_nan"),
    ]

    for col, mask, step_name in clean_rules:
        if col not in df.columns:
            continue
        bad = int(mask.sum())
        if bad > 0:
            df.loc[mask, col] = np.nan
        report_rows.append({"step": step_name, "value": bad})

    numeric_all = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_wo_target = [c for c in numeric_all if c != "default_flag"]

    num_missing_before = int(df[numeric_wo_target].isna().sum().sum()) if numeric_wo_target else 0
    cat_missing_before = int(df[categorical_cols].isna().sum().sum()) if categorical_cols else 0

    for c in numeric_wo_target:
        med = df[c].median()
        df[c] = df[c].fillna(0 if pd.isna(med) else med)

    for c in categorical_cols:
        df[c] = df[c].fillna("Unknown").astype(str)

    num_missing_after = int(df[numeric_wo_target].isna().sum().sum()) if numeric_wo_target else 0
    cat_missing_after = int(df[categorical_cols].isna().sum().sum()) if categorical_cols else 0

    report_rows.append({"step": "numeric_missing_before", "value": num_missing_before})
    report_rows.append({"step": "numeric_missing_after", "value": num_missing_after})
    report_rows.append({"step": "categorical_missing_before", "value": cat_missing_before})
    report_rows.append({"step": "categorical_missing_after", "value": cat_missing_after})
    report_rows.append({"step": "final_shape", "value": f"{len(df)}x{len(df.columns)}"})
    report_rows.append({"step": "default_rate", "value": round(float(df["default_flag"].mean()), 6)})

    missing_by_col = df.isna().sum().reset_index()
    missing_by_col.columns = ["column", "missing_count"]
    missing_by_col["missing_ratio"] = (missing_by_col["missing_count"] / len(df)).round(6)

    report_df = pd.DataFrame(report_rows)
    return df, report_df, missing_by_col


def main():
    parser = argparse.ArgumentParser(description="Clean credit risk dataset for modeling.")
    parser.add_argument(
        "--input",
        default="credit_risk_dataset.csv/credit_risk_dataset.csv",
        help="Raw CSV path",
    )
    parser.add_argument("--out-clean", default="data/processed/loan_clean.csv", help="Cleaned CSV path")
    parser.add_argument("--out-report", default="data/processed/clean_report.csv", help="Cleaning report CSV path")
    parser.add_argument("--out-missing", default="data/processed/missing_by_column.csv", help="Column-missing report CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    clean_df, report_df, missing_df = clean_credit_risk_data(df)

    os.makedirs(os.path.dirname(args.out_clean), exist_ok=True)
    clean_df.to_csv(args.out_clean, index=False)
    report_df.to_csv(args.out_report, index=False)
    missing_df.to_csv(args.out_missing, index=False)

    print("Done.")
    print(f"Cleaned file   : {args.out_clean}")
    print(f"Report file    : {args.out_report}")
    print(f"Missing detail : {args.out_missing}")
    print(f"Final rows     : {len(clean_df)}")
    print(f"Default rate   : {clean_df['default_flag'].mean():.2%}")


if __name__ == "__main__":
    main()
