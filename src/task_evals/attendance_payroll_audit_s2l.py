import json
from pathlib import Path
from typing import Any, Optional

import dspy
import pandas as pd

from src.task_setups.attendance_payroll_audit_s2l import OUTPUT_FILENAME, build_expected_output


def _load_expected(path: Path) -> pd.DataFrame:
    example = json.loads(path.read_text(encoding="utf-8"))
    attendance_df = pd.DataFrame(example["attendance_rows"])
    assignments_df = pd.DataFrame(example["department_assignments"])
    rates_df = pd.DataFrame(example["rates"])
    return build_expected_output(attendance_df, assignments_df, rates_df)


def _load_output(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(path)
    except Exception:
        return None


def _standardize(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    standardized = df.copy()
    standardized.columns = (
        standardized.columns.str.strip().str.lower().str.replace(" ", "").str.replace("-", "")
    )
    if "department" in standardized.columns:
        standardized["department"] = standardized["department"].astype(str).str.strip()
        standardized = standardized.sort_values("department").reset_index(drop=True)
    return standardized


def _columns_correct(output_df: Optional[pd.DataFrame]) -> bool:
    if output_df is None:
        return False
    expected_columns = {
        "department",
        "departmentaverageworklength",
        "departmentaverageontimedeparturecount",
        "departmentaveragelatearrivalcount",
        "departmenttotalpayroll",
    }
    return set(output_df.columns) == expected_columns


def _values_correct(output_df: Optional[pd.DataFrame], expected_df: pd.DataFrame) -> bool:
    if output_df is None:
        return False
    if set(output_df["department"]) != set(expected_df["department"]):
        return False

    merged = output_df.merge(expected_df, on="department", suffixes=("_output", "_expected"))
    numeric_columns = [
        "departmentaverageworklength",
        "departmentaverageontimedeparturecount",
        "departmentaveragelatearrivalcount",
        "departmenttotalpayroll",
    ]
    for column in numeric_columns:
        diff = (
            merged[f"{column}_output"].astype(float) - merged[f"{column}_expected"].astype(float)
        ).abs()
        tolerance = 1e-5 if column != "departmenttotalpayroll" else 1e-2
        if not (diff <= tolerance).all():
            return False
    return True


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> dict[str, Any]:
    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    output_path = workspace / OUTPUT_FILENAME
    expected_path = log_dir / "groundtruth" / "example.json"

    output_df = _standardize(_load_output(output_path))
    expected_df = _standardize(_load_expected(expected_path))

    checkpoints: list[tuple[str, bool, int]] = [
        ("created Excel report", output_path.exists(), 1),
        ("used exact output columns", _columns_correct(output_df), 1),
        ("matched expected department aggregates", _values_correct(output_df, expected_df), 2),
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
