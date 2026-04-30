import json
from pathlib import Path
from typing import Any, Optional

import dspy
import pandas as pd

from src.task_setups.expense_reconciliation_s2l import (
    EXPENSES_OUTPUT_FILENAME,
    ANALYSIS_OUTPUT_FILENAME,
    FLAGGED_PAYMENTS_OUTPUT_FILENAME,
    build_expected_analysis,
    build_expected_expenses_corrected,
)


def _load_excel(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(path)
    except Exception:
        return None


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _standardize_columns(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "").str.replace("_", "")
    return df


def _normalized_string_series(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in df.columns:
        return None
    series = df[column]
    if series.isna().any():
        return None
    return series.astype(str).str.strip()


def _categories_correct(output_df: Optional[pd.DataFrame], expected_df: pd.DataFrame) -> bool:
    if output_df is None:
        return False
    out = _standardize_columns(output_df)
    exp = _standardize_columns(expected_df)
    if "correctcategory" not in out.columns:
        return False
    if len(out) != len(exp):
        return False
    out_categories = _normalized_string_series(out, "correctcategory")
    exp_categories = _normalized_string_series(exp, "correctcategory")
    if out_categories is None or exp_categories is None:
        return False
    return (out_categories.values == exp_categories.values).all()


def _analysis_columns_correct(output_df: Optional[pd.DataFrame]) -> bool:
    if output_df is None:
        return False
    out = _standardize_columns(output_df)
    expected_cols = {"category", "totalamount", "numberofemployees", "costperemployee"}
    return expected_cols.issubset(set(out.columns))


def _analysis_values_correct(output_df: Optional[pd.DataFrame], expected_df: pd.DataFrame) -> bool:
    if output_df is None:
        return False
    out = _standardize_columns(output_df)
    exp = _standardize_columns(expected_df)
    if not _analysis_columns_correct(output_df):
        return False
    out_categories = _normalized_string_series(out, "category")
    exp_categories = _normalized_string_series(exp, "category")
    if out_categories is None or exp_categories is None:
        return False
    if set(out_categories) != set(exp_categories):
        return False
    merged = out.merge(
        exp.rename(columns=lambda c: c + "_exp"),
        left_on="category",
        right_on="category_exp",
        how="inner",
    )
    for col in ["totalamount", "numberofemployees", "costperemployee"]:
        diff = (merged[col].astype(float) - merged[col + "_exp"].astype(float)).abs()
        if not (diff <= 0.02).all():
            return False
    return True


def _flagged_payments_correct(output_df: Optional[pd.DataFrame], expected_issues: list[dict]) -> bool:
    if output_df is None:
        return False
    out = _standardize_columns(output_df)
    if "paymentid" not in out.columns or "issue" not in out.columns:
        return False
    exp_ids = {str(row["Payment_ID"]) for row in expected_issues}
    out_ids = set(out["paymentid"].astype(str).str.strip())
    return exp_ids == out_ids


def _total_row_correct(output_df: Optional[pd.DataFrame], expected_issues: list[dict]) -> bool:
    if output_df is None:
        return False
    out = _standardize_columns(output_df)
    if "paymentid" not in out.columns or "issue" not in out.columns:
        return False
    total_rows = out[out["paymentid"].astype(str).str.strip() == "TOTAL"]
    if total_rows.empty:
        return False
    total_issue = total_rows.iloc[-1]["issue"]
    expected_total = next(
        (r["Issue"] for r in expected_issues if str(r["Payment_ID"]) == "TOTAL"), None
    )
    if expected_total is None:
        return False
    return str(total_issue).strip() == expected_total.strip()


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> dict[str, Any]:
    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    groundtruth_dir = log_dir / "groundtruth"

    expenses_corrected_path = workspace / EXPENSES_OUTPUT_FILENAME
    analysis_path = workspace / ANALYSIS_OUTPUT_FILENAME
    flagged_path = workspace / FLAGGED_PAYMENTS_OUTPUT_FILENAME

    expected_corrected_df = _load_excel(groundtruth_dir / "expenses_corrected.xlsx")
    expected_analysis_df = _load_csv(groundtruth_dir / "expense_analysis.csv")
    expected_issues = example["issues"]

    output_corrected_df = _load_excel(expenses_corrected_path)
    output_analysis_df = _load_csv(analysis_path)
    output_flagged_df = _load_excel(flagged_path)

    checkpoints: list[tuple[str, bool, int]] = [
        (
            "corrected expense categories",
            _categories_correct(output_corrected_df, expected_corrected_df),
            2,
        ),
        (
            "expense analysis columns",
            _analysis_columns_correct(output_analysis_df),
            1,
        ),
        (
            "expense analysis values",
            _analysis_values_correct(output_analysis_df, expected_analysis_df),
            2,
        ),
        (
            "flagged all anomalous payments",
            _flagged_payments_correct(output_flagged_df, expected_issues),
            1,
        ),
        (
            "correct TOTAL summary row",
            _total_row_correct(output_flagged_df, expected_issues),
            1,
        ),
    ]

    total = sum(weight for _, _, weight in checkpoints)
    score = sum(weight for _, passed, weight in checkpoints if passed)
    if checkpoints[-1][1]:
        score = total

    failed = [name for name, passed, _ in checkpoints if not passed]
    feedback = "All checkpoints passed." if not failed else "Failed checkpoints: " + ", ".join(failed)

    return {
        "workspace_dir": workspace_dir,
        "score": score / total,
        "feedback": feedback,
        "example_id": example.get("id"),
    }
