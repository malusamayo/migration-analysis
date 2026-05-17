"""Compositional atom library for attendance-payroll-audit task variants.

Templates are declarative compositions over three atom categories:

- DOCUMENTS: which reference files appear in the workspace
- COMPUTATIONS: how the per-employee work-hours / on-time / late / payroll
  aggregates are derived (varies with overtime, holiday, shift, probation,
  tax, bonus, or aggregation-level policies)
- ACTIONS: which deliverable artifacts the agent must produce. The primary
  deliverable is always an aggregated audit table (Excel/CSV/JSON); some
  templates additionally require exceptions.txt, summary.md, top_late.txt,
  or violations.txt.

All variants share a common attendance-log structure (date, clock-in,
clock-out, employee name); the policies and deliverables differ.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import dspy
import pandas as pd


# ---------------------------------------------------------------------------
# Document paths
# ---------------------------------------------------------------------------

ATTENDANCE_DIR = Path("Documents/Human Resources Team/Attendance")

DOC_PATHS = {
    "attendance": ATTENDANCE_DIR / "april-attendance-data.csv",
    "departments": ATTENDANCE_DIR / "employee-department.txt",
    "rates": ATTENDANCE_DIR / "salary-rates.txt",
    "overtime_policy": ATTENDANCE_DIR / "overtime-policy.md",
    "holidays": ATTENDANCE_DIR / "holidays.csv",
    "probation": ATTENDANCE_DIR / "probation-list.txt",
    "shift_schedule": ATTENDANCE_DIR / "shift-schedule.csv",
    "teams": ATTENDANCE_DIR / "team-mapping.txt",
    "roles": ATTENDANCE_DIR / "role-mapping.txt",
    "tax_table": ATTENDANCE_DIR / "tax-table.csv",
    "bonus_policy": ATTENDANCE_DIR / "bonus-policy.md",
    "minimum_wage": ATTENDANCE_DIR / "minimum-wage-policy.md",
}


# ---------------------------------------------------------------------------
# Document builders
# ---------------------------------------------------------------------------


def build_attendance_doc(example: dict) -> str:
    df = pd.DataFrame(example["attendance_rows"])
    return df.to_csv(index=False).rstrip("\n")


def build_departments_doc(example: dict) -> str:
    assignments = pd.DataFrame(example["department_assignments"])
    lines = []
    grouped = assignments.groupby("Department")["Name"].apply(list).reset_index()
    for row in grouped.sort_values("Department").itertuples(index=False):
        lines.append(f"• {row.Department}: {', '.join(row.Name)}")
    return "\n".join(lines)


def build_rates_doc(example: dict) -> str:
    rates = sorted(example["rates"], key=lambda r: r["Name"])
    return "\n".join(f'{r["Name"]}: {r["Rate"]:.2f}' for r in rates)


def build_overtime_policy_doc(example: dict) -> str:
    return (
        "# Overtime policy\n\n"
        "Hours worked up to 8 hours on a single day are paid at the employee's "
        "standard hourly rate. Any additional hours on that same day are paid at "
        "1.5x the standard rate. Total payroll is the sum across days of "
        "`min(daily_hours, 8) * rate + max(0, daily_hours - 8) * 1.5 * rate`, "
        "with the per-employee total rounded to two decimals. No ceiling "
        "rounding is applied."
    )


def build_holidays_doc(example: dict) -> str:
    rows = sorted(example["holidays"], key=lambda r: r["Date"])
    lines = ["date,name"]
    for row in rows:
        lines.append(f'{row["Date"]},{row["Name"]}')
    return "\n".join(lines)


def build_probation_doc(example: dict) -> str:
    names = sorted(example["probation"])
    header = (
        "Employees currently on probation. Exclude these employees from "
        "department-level aggregates and from total payroll."
    )
    return header + "\n\n" + "\n".join(f"- {n}" for n in names)


def build_shift_schedule_doc(example: dict) -> str:
    rows = sorted(example["shifts"], key=lambda r: r["Name"])
    lines = ["name,expected_clock_in,expected_clock_out"]
    for row in rows:
        lines.append(f'{row["Name"]},{row["Expected_Clock_In"]},{row["Expected_Clock_Out"]}')
    return "\n".join(lines)


def build_teams_doc(example: dict) -> str:
    df = pd.DataFrame(example["team_assignments"])
    lines = []
    grouped = df.groupby("Team")["Name"].apply(list).reset_index()
    for row in grouped.sort_values("Team").itertuples(index=False):
        lines.append(f"• {row.Team}: {', '.join(row.Name)}")
    return "\n".join(lines)


def build_roles_doc(example: dict) -> str:
    df = pd.DataFrame(example["role_assignments"])
    rows = df.sort_values("Name")
    return "\n".join(f'{row.Name}: {row.Role}' for row in rows.itertuples(index=False))


def build_tax_table_doc(example: dict) -> str:
    table = example["tax_table"]
    lines = ["bracket_max_payroll,tax_rate"]
    for row in table:
        ceiling = "inf" if row["Max"] is None else f'{row["Max"]:.2f}'
        lines.append(f'{ceiling},{row["Rate"]:.2f}')
    return "\n".join(lines)


def build_bonus_policy_doc(example: dict) -> str:
    pol = example["bonus_policy"]
    return (
        "# Bonus policy\n\n"
        f"Any employee whose on-time departure count is at least "
        f"{pol['on_time_threshold']} days receives a bonus equal to "
        f"{int(round(pol['bonus_pct'] * 100))}% of their total payroll. "
        "The department bonus is the sum of qualifying employees' bonuses; "
        "round each employee's bonus to two decimals before summing."
    )


def build_minimum_wage_doc(example: dict) -> str:
    threshold = example["minimum_wage_threshold"]
    return (
        "# Minimum wage policy\n\n"
        f"Every employee's hourly rate must be at least {threshold:.2f}. Any "
        "employee paid below that rate is a violation and must be listed in "
        "the audit's violations file."
    )


DOC_BUILDERS: dict[str, Callable[[dict], str]] = {
    "attendance": build_attendance_doc,
    "departments": build_departments_doc,
    "rates": build_rates_doc,
    "overtime_policy": build_overtime_policy_doc,
    "holidays": build_holidays_doc,
    "probation": build_probation_doc,
    "shift_schedule": build_shift_schedule_doc,
    "teams": build_teams_doc,
    "roles": build_roles_doc,
    "tax_table": build_tax_table_doc,
    "bonus_policy": build_bonus_policy_doc,
    "minimum_wage": build_minimum_wage_doc,
}


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def _parse_time(text: str) -> int:
    """Return minutes since midnight for HH:MM."""
    hour, minute = text.split(":")
    return int(hour) * 60 + int(minute)


def _attendance_df(example: dict) -> pd.DataFrame:
    df = pd.DataFrame(example["attendance_rows"]).copy()
    df["clock_in_min"] = df["Clock-in"].map(_parse_time)
    df["clock_out_min"] = df["Clock-out"].map(_parse_time)
    df["work_minutes"] = df["clock_out_min"] - df["clock_in_min"]
    df["work_hours"] = df["work_minutes"] / 60.0
    return df


def _basic_per_employee(
    attendance: pd.DataFrame,
    rates: dict[str, float],
    *,
    late_after: int = _parse_time("09:00"),
    on_time_window: tuple[int, int] = (_parse_time("17:30"), _parse_time("18:00")),
) -> pd.DataFrame:
    df = attendance.copy()
    df["on_time"] = df["clock_out_min"].between(on_time_window[0], on_time_window[1])
    df["late"] = df["clock_in_min"] > late_after
    grouped = (
        df.groupby("Name")
        .agg(
            avg_work_length=("work_hours", "mean"),
            on_time_count=("on_time", "sum"),
            late_count=("late", "sum"),
            total_work_hours=("work_hours", "sum"),
        )
        .reset_index()
    )
    grouped["rate"] = grouped["Name"].map(rates)
    grouped["total_payroll"] = grouped["total_work_hours"].apply(math.ceil) * grouped["rate"]
    grouped["total_payroll"] = grouped["total_payroll"].round(2)
    return grouped


def _rates_map(example: dict) -> dict[str, float]:
    return {row["Name"]: float(row["Rate"]) for row in example["rates"]}


def _dept_map(example: dict) -> dict[str, str]:
    return {row["Name"]: row["Department"] for row in example["department_assignments"]}


def _team_map(example: dict) -> dict[str, str]:
    return {row["Name"]: row["Team"] for row in example["team_assignments"]}


def _role_map(example: dict) -> dict[str, str]:
    return {row["Name"]: row["Role"] for row in example["role_assignments"]}


def _aggregate_by(
    employee_df: pd.DataFrame,
    group_col: str,
    payroll_col: str = "total_payroll",
    extra_sum_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    extra_sum_cols = extra_sum_cols or []
    agg_kwargs = {
        f"{group_col} Average Work Length": ("avg_work_length", "mean"),
        f"{group_col} Average On-time Departure Count": ("on_time_count", "mean"),
        f"{group_col} Average Late Arrival Count": ("late_count", "mean"),
        f"{group_col} Total Payroll": (payroll_col, "sum"),
    }
    for col in extra_sum_cols:
        agg_kwargs[col] = (col, "sum")
    result = employee_df.groupby(group_col).agg(**agg_kwargs).reset_index()
    result = result.sort_values(group_col).reset_index(drop=True)
    for col in result.columns:
        if col == group_col:
            continue
        result[col] = result[col].astype(float).round(4 if "Average" in col else 2)
    return result


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------


def compute_basic_dept(example: dict) -> dict:
    attendance = _attendance_df(example)
    rates = _rates_map(example)
    employees = _basic_per_employee(attendance, rates)
    employees["Department"] = employees["Name"].map(_dept_map(example))
    result = _aggregate_by(employees, "Department")
    return {"primary_df": result, "primary_group": "Department", "employees": employees}


def compute_overtime_dept(example: dict) -> dict:
    attendance = _attendance_df(example)
    rates = _rates_map(example)
    df = attendance.copy()
    df["rate"] = df["Name"].map(rates)
    df["std_hours"] = df["work_hours"].clip(upper=8)
    df["ot_hours"] = (df["work_hours"] - 8).clip(lower=0)
    df["daily_pay"] = df["std_hours"] * df["rate"] + df["ot_hours"] * df["rate"] * 1.5
    df["on_time"] = df["clock_out_min"].between(_parse_time("17:30"), _parse_time("18:00"))
    df["late"] = df["clock_in_min"] > _parse_time("09:00")
    employees = (
        df.groupby("Name")
        .agg(
            avg_work_length=("work_hours", "mean"),
            on_time_count=("on_time", "sum"),
            late_count=("late", "sum"),
            total_payroll=("daily_pay", "sum"),
        )
        .reset_index()
    )
    employees["total_payroll"] = employees["total_payroll"].round(2)
    employees["Department"] = employees["Name"].map(_dept_map(example))
    result = _aggregate_by(employees, "Department")
    return {"primary_df": result, "primary_group": "Department", "employees": employees}


def compute_holiday_adj_dept(example: dict) -> dict:
    attendance = _attendance_df(example)
    rates = _rates_map(example)
    holiday_dates = {row["Date"] for row in example["holidays"]}
    df = attendance.copy()
    is_holiday = df["Date"].isin(holiday_dates)
    df["on_time"] = (~is_holiday) & df["clock_out_min"].between(
        _parse_time("17:30"), _parse_time("18:00")
    )
    df["late"] = (~is_holiday) & (df["clock_in_min"] > _parse_time("09:00"))
    employees = (
        df.groupby("Name")
        .agg(
            avg_work_length=("work_hours", "mean"),
            on_time_count=("on_time", "sum"),
            late_count=("late", "sum"),
            total_work_hours=("work_hours", "sum"),
        )
        .reset_index()
    )
    employees["rate"] = employees["Name"].map(rates)
    employees["total_payroll"] = (
        employees["total_work_hours"].apply(math.ceil) * employees["rate"]
    ).round(2)
    employees["Department"] = employees["Name"].map(_dept_map(example))
    result = _aggregate_by(employees, "Department")
    return {"primary_df": result, "primary_group": "Department", "employees": employees}


def compute_probation_dept(example: dict) -> dict:
    base = compute_basic_dept(example)
    probation = set(example["probation"])
    employees = base["employees"]
    kept = employees[~employees["Name"].isin(probation)].reset_index(drop=True)
    result = _aggregate_by(kept, "Department")
    return {"primary_df": result, "primary_group": "Department", "employees": kept}


def compute_shift_dept(example: dict) -> dict:
    attendance = _attendance_df(example)
    rates = _rates_map(example)
    shift_map = {
        row["Name"]: (
            _parse_time(row["Expected_Clock_In"]),
            _parse_time(row["Expected_Clock_Out"]),
        )
        for row in example["shifts"]
    }
    df = attendance.copy()
    expected_out = df["Name"].map({n: v[1] for n, v in shift_map.items()})
    expected_in = df["Name"].map({n: v[0] for n, v in shift_map.items()})
    df["on_time"] = (df["clock_out_min"] - expected_out).abs() <= 15
    df["late"] = df["clock_in_min"] > expected_in
    employees = (
        df.groupby("Name")
        .agg(
            avg_work_length=("work_hours", "mean"),
            on_time_count=("on_time", "sum"),
            late_count=("late", "sum"),
            total_work_hours=("work_hours", "sum"),
        )
        .reset_index()
    )
    employees["rate"] = employees["Name"].map(rates)
    employees["total_payroll"] = (
        employees["total_work_hours"].apply(math.ceil) * employees["rate"]
    ).round(2)
    employees["Department"] = employees["Name"].map(_dept_map(example))
    result = _aggregate_by(employees, "Department")
    return {"primary_df": result, "primary_group": "Department", "employees": employees}


def compute_bonus_dept(example: dict) -> dict:
    base = compute_basic_dept(example)
    employees = base["employees"].copy()
    pol = example["bonus_policy"]
    threshold = int(pol["on_time_threshold"])
    pct = float(pol["bonus_pct"])
    employees["bonus"] = 0.0
    qualifies = employees["on_time_count"] >= threshold
    employees.loc[qualifies, "bonus"] = (
        employees.loc[qualifies, "total_payroll"] * pct
    ).round(2)
    result = _aggregate_by(
        employees,
        "Department",
        extra_sum_cols=["bonus"],
    )
    result = result.rename(columns={"bonus": "Department Total Bonus"})
    return {"primary_df": result, "primary_group": "Department", "employees": employees}


def compute_tax_dept(example: dict) -> dict:
    base = compute_basic_dept(example)
    employees = base["employees"].copy()
    table = sorted(example["tax_table"], key=lambda r: float("inf") if r["Max"] is None else r["Max"])

    def tax_rate(amount: float) -> float:
        for row in table:
            ceiling = float("inf") if row["Max"] is None else row["Max"]
            if amount <= ceiling:
                return float(row["Rate"])
        return float(table[-1]["Rate"])

    employees["tax_rate"] = employees["total_payroll"].apply(tax_rate)
    employees["net_payroll"] = (
        employees["total_payroll"] * (1 - employees["tax_rate"])
    ).round(2)
    result = _aggregate_by(
        employees,
        "Department",
        extra_sum_cols=["net_payroll"],
    )
    result = result.rename(columns={"net_payroll": "Department Net Payroll"})
    return {"primary_df": result, "primary_group": "Department", "employees": employees}


def compute_team_agg(example: dict) -> dict:
    base = compute_basic_dept(example)
    employees = base["employees"].copy()
    employees["Team"] = employees["Name"].map(_team_map(example))
    result = _aggregate_by(employees, "Team")
    return {"primary_df": result, "primary_group": "Team", "employees": employees}


def compute_role_agg(example: dict) -> dict:
    base = compute_basic_dept(example)
    employees = base["employees"].copy()
    employees["Role"] = employees["Name"].map(_role_map(example))
    result = _aggregate_by(employees, "Role")
    return {"primary_df": result, "primary_group": "Role", "employees": employees}


def compute_overtime_holiday_dept(example: dict) -> dict:
    attendance = _attendance_df(example)
    rates = _rates_map(example)
    holiday_dates = {row["Date"] for row in example["holidays"]}
    df = attendance.copy()
    df["rate"] = df["Name"].map(rates)
    df["std_hours"] = df["work_hours"].clip(upper=8)
    df["ot_hours"] = (df["work_hours"] - 8).clip(lower=0)
    df["daily_pay"] = df["std_hours"] * df["rate"] + df["ot_hours"] * df["rate"] * 1.5
    is_holiday = df["Date"].isin(holiday_dates)
    df["on_time"] = (~is_holiday) & df["clock_out_min"].between(
        _parse_time("17:30"), _parse_time("18:00")
    )
    df["late"] = (~is_holiday) & (df["clock_in_min"] > _parse_time("09:00"))
    employees = (
        df.groupby("Name")
        .agg(
            avg_work_length=("work_hours", "mean"),
            on_time_count=("on_time", "sum"),
            late_count=("late", "sum"),
            total_payroll=("daily_pay", "sum"),
        )
        .reset_index()
    )
    employees["total_payroll"] = employees["total_payroll"].round(2)
    employees["Department"] = employees["Name"].map(_dept_map(example))
    result = _aggregate_by(employees, "Department")
    return {"primary_df": result, "primary_group": "Department", "employees": employees}


def compute_probation_overtime_dept(example: dict) -> dict:
    full = compute_overtime_dept(example)
    employees = full["employees"]
    probation = set(example["probation"])
    kept = employees[~employees["Name"].isin(probation)].reset_index(drop=True)
    result = _aggregate_by(kept, "Department")
    return {"primary_df": result, "primary_group": "Department", "employees": kept}


COMPUTATIONS: dict[str, Callable[[dict], dict]] = {
    "basic_dept": compute_basic_dept,
    "overtime_dept": compute_overtime_dept,
    "holiday_adj_dept": compute_holiday_adj_dept,
    "probation_dept": compute_probation_dept,
    "shift_dept": compute_shift_dept,
    "bonus_dept": compute_bonus_dept,
    "tax_dept": compute_tax_dept,
    "team_agg": compute_team_agg,
    "role_agg": compute_role_agg,
    "overtime_holiday_dept": compute_overtime_holiday_dept,
    "probation_overtime_dept": compute_probation_overtime_dept,
}


# ---------------------------------------------------------------------------
# Secondary deliverables (derived from `employees` per-employee frame)
# ---------------------------------------------------------------------------


LATE_EXCEPTION_THRESHOLD = 3
TOP_LATE_K = 3


def _expected_exceptions(employees: pd.DataFrame) -> list[str]:
    rows = employees[employees["late_count"] >= LATE_EXCEPTION_THRESHOLD]
    return sorted(rows["Name"].tolist())


def _expected_top_late(employees: pd.DataFrame) -> list[str]:
    sorted_df = employees.sort_values(
        ["late_count", "Name"], ascending=[False, True]
    )
    return sorted_df.head(TOP_LATE_K)["Name"].tolist()


def _expected_violations(example: dict) -> list[str]:
    threshold = float(example["minimum_wage_threshold"])
    return sorted(
        row["Name"] for row in example["rates"] if float(row["Rate"]) < threshold
    )


def _expected_summary_terms(primary_df: pd.DataFrame, group_col: str) -> list[str]:
    return [str(value) for value in primary_df[group_col].tolist()]


# ---------------------------------------------------------------------------
# Action checks
# ---------------------------------------------------------------------------


def _standardize_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    standardized = df.copy()
    standardized.columns = [
        re.sub(r"[\s\-]+", "", str(col).strip().lower()) for col in standardized.columns
    ]
    key = group_col.lower()
    if key in standardized.columns:
        standardized[key] = standardized[key].astype(str).str.strip()
        standardized = standardized.sort_values(key).reset_index(drop=True)
    return standardized


def _expected_standardized(primary_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return _standardize_table(primary_df, group_col)


def _values_match(actual: pd.DataFrame, expected: pd.DataFrame, group_col: str) -> bool:
    key = group_col.lower()
    if key not in actual.columns or key not in expected.columns:
        return False
    if set(actual[key]) != set(expected[key]):
        return False
    merged = actual.merge(expected, on=key, suffixes=("_actual", "_expected"))
    numeric_cols = [c for c in expected.columns if c != key]
    for col in numeric_cols:
        if f"{col}_actual" not in merged.columns:
            return False
        diff = (
            merged[f"{col}_actual"].astype(float)
            - merged[f"{col}_expected"].astype(float)
        ).abs()
        tolerance = 1.1e-2 if "payroll" in col or "bonus" in col else 5.1e-3
        if not (diff <= tolerance).all():
            return False
    return True


def _columns_match(actual: pd.DataFrame, expected: pd.DataFrame) -> bool:
    return set(actual.columns) == set(expected.columns)


def _read_table(path: Path, fmt: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        if fmt == "xlsx":
            return pd.read_excel(path)
        if fmt == "csv":
            return pd.read_csv(path)
        if fmt == "json":
            return pd.read_json(path)
    except Exception:
        return None
    return None


def _check_table_file(
    state: dict,
    expected: dict,
    workspace: Path,
    lm: Optional[dspy.LM],
    *,
    filename: str,
    fmt: str,
) -> bool:
    group_col = expected["primary_group"]
    actual = _read_table(workspace / filename, fmt)
    if actual is None:
        return False
    actual_std = _standardize_table(actual, group_col)
    expected_df = pd.DataFrame(expected["primary_df_records"])
    expected_std = _expected_standardized(expected_df, group_col)
    if not _columns_match(actual_std, expected_std):
        return False
    return _values_match(actual_std, expected_std, group_col)


def check_write_xlsx_basic(state, expected, workspace, lm):
    return _check_table_file(
        state, expected, workspace, lm,
        filename="attendance-payroll-audit.xlsx", fmt="xlsx",
    )


def check_write_csv_basic(state, expected, workspace, lm):
    return _check_table_file(
        state, expected, workspace, lm,
        filename="attendance-payroll-audit.csv", fmt="csv",
    )


def check_write_json_basic(state, expected, workspace, lm):
    return _check_table_file(
        state, expected, workspace, lm,
        filename="attendance-payroll-audit.json", fmt="json",
    )


def check_write_xlsx_with_bonus(state, expected, workspace, lm):
    return _check_table_file(
        state, expected, workspace, lm,
        filename="attendance-payroll-audit.xlsx", fmt="xlsx",
    )


def check_write_xlsx_net(state, expected, workspace, lm):
    return _check_table_file(
        state, expected, workspace, lm,
        filename="attendance-payroll-audit.xlsx", fmt="xlsx",
    )


def check_write_xlsx_team(state, expected, workspace, lm):
    return _check_table_file(
        state, expected, workspace, lm,
        filename="team-payroll-audit.xlsx", fmt="xlsx",
    )


def check_write_xlsx_role(state, expected, workspace, lm):
    return _check_table_file(
        state, expected, workspace, lm,
        filename="role-payroll-audit.xlsx", fmt="xlsx",
    )


def _read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _extract_names(text: str, candidates: set[str]) -> set[str]:
    found = set()
    for name in candidates:
        if name in text:
            found.add(name)
    return found


def check_write_exceptions(state, expected, workspace, lm):
    text = _read_text(workspace / "exceptions.txt")
    if text is None:
        return False
    target = set(expected["expected_exceptions"])
    all_names = set(expected["all_employee_names"])
    mentioned = _extract_names(text, all_names)
    return mentioned == target


def check_write_summary(state, expected, workspace, lm):
    text = _read_text(workspace / "summary.md")
    if text is None:
        return False
    for term in expected["expected_summary_terms"]:
        if term not in text:
            return False
    return True


def check_write_top_late(state, expected, workspace, lm):
    text = _read_text(workspace / "top_late.txt")
    if text is None:
        return False
    target = list(expected["expected_top_late"])
    all_names = set(expected["all_employee_names"])
    mentioned = _extract_names(text, all_names)
    return mentioned == set(target)


def check_write_violations(state, expected, workspace, lm):
    text = _read_text(workspace / "violations.txt")
    if text is None:
        return False
    target = set(expected["expected_violations"])
    all_names = set(expected["all_employee_names"])
    mentioned = _extract_names(text, all_names)
    return mentioned == target


@dataclass
class ActionSpec:
    name: str
    weight: int
    check: Callable[[dict, dict, Path, Optional[dspy.LM]], bool]
    terminal: bool = False


ACTIONS: dict[str, ActionSpec] = {
    "write_xlsx_basic": ActionSpec(
        name="wrote attendance-payroll-audit.xlsx with correct dept aggregates",
        weight=2,
        check=check_write_xlsx_basic,
        terminal=True,
    ),
    "write_csv_basic": ActionSpec(
        name="wrote attendance-payroll-audit.csv with correct dept aggregates",
        weight=2,
        check=check_write_csv_basic,
        terminal=True,
    ),
    "write_json_basic": ActionSpec(
        name="wrote attendance-payroll-audit.json with correct dept aggregates",
        weight=2,
        check=check_write_json_basic,
        terminal=True,
    ),
    "write_xlsx_with_bonus": ActionSpec(
        name="wrote attendance-payroll-audit.xlsx with bonus column",
        weight=2,
        check=check_write_xlsx_with_bonus,
        terminal=True,
    ),
    "write_xlsx_net": ActionSpec(
        name="wrote attendance-payroll-audit.xlsx with net payroll column",
        weight=2,
        check=check_write_xlsx_net,
        terminal=True,
    ),
    "write_xlsx_team": ActionSpec(
        name="wrote team-payroll-audit.xlsx with team aggregates",
        weight=2,
        check=check_write_xlsx_team,
        terminal=True,
    ),
    "write_xlsx_role": ActionSpec(
        name="wrote role-payroll-audit.xlsx with role aggregates",
        weight=2,
        check=check_write_xlsx_role,
        terminal=True,
    ),
    "write_exceptions": ActionSpec(
        name="wrote exceptions.txt with chronic late employees",
        weight=1,
        check=check_write_exceptions,
    ),
    "write_summary": ActionSpec(
        name="wrote summary.md naming each group",
        weight=1,
        check=check_write_summary,
    ),
    "write_top_late": ActionSpec(
        name="wrote top_late.txt with the top late employees",
        weight=1,
        check=check_write_top_late,
    ),
    "write_violations": ActionSpec(
        name="wrote violations.txt with sub-minimum-wage employees",
        weight=1,
        check=check_write_violations,
    ),
}


# ---------------------------------------------------------------------------
# Expected outcome builder
# ---------------------------------------------------------------------------


def required_documents(template_docs: list[str]) -> list[str]:
    return [str(DOC_PATHS[d]) for d in template_docs]


def build_expected(example: dict, template: dict) -> dict:
    compute = COMPUTATIONS[template["compute"]]
    result = compute(example)
    primary_df: pd.DataFrame = result["primary_df"]
    employees: pd.DataFrame = result["employees"]
    expected: dict[str, Any] = {
        "template_id": template["id"],
        "required_documents": required_documents(template["docs"]),
        "primary_group": result["primary_group"],
        "primary_df_records": primary_df.to_dict(orient="records"),
        "primary_columns": list(primary_df.columns),
        "all_employee_names": sorted(employees["Name"].tolist()),
    }

    actions = template["actions"]
    if "write_exceptions" in actions:
        expected["expected_exceptions"] = _expected_exceptions(employees)
    if "write_summary" in actions:
        expected["expected_summary_terms"] = _expected_summary_terms(
            primary_df, result["primary_group"]
        )
    if "write_top_late" in actions:
        expected["expected_top_late"] = _expected_top_late(employees)
    if "write_violations" in actions:
        expected["expected_violations"] = _expected_violations(example)

    return expected
