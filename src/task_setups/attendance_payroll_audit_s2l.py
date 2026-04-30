"""Workspace setup for the attendance_payroll_audit_s2l task."""

import json
import math
import random
from pathlib import Path

import pandas as pd


ATTENDANCE_DIR = Path("Documents/Human Resources Team/Attendance")
ATTENDANCE_DOC_PATH = ATTENDANCE_DIR / "april-attendance-data.csv"
DEPARTMENT_DOC_PATH = ATTENDANCE_DIR / "employee-department.txt"
RATES_DOC_PATH = ATTENDANCE_DIR / "salary-rates.txt"
OUTPUT_FILENAME = "attendance-payroll-audit.xlsx"

DEPARTMENT_POOL = [
    "Finance",
    "HR",
    "Operations",
    "Sales",
    "Engineering",
    "Product",
    "Support",
    "Legal",
]

NAME_POOL = [
    "Ava Thompson",
    "Benjamin Lee",
    "Charlotte Davis",
    "Daniel Walker",
    "Ella Martin",
    "Felix Young",
    "Grace Hall",
    "Henry Allen",
    "Isla Scott",
    "Jack Green",
    "Katherine Baker",
    "Liam Adams",
    "Mia Nelson",
    "Noah Carter",
    "Olivia Perez",
    "Patrick Roberts",
    "Quinn Turner",
    "Ruby Phillips",
    "Samuel Campbell",
    "Tessa Parker",
    "Uma Evans",
    "Victor Edwards",
    "Willow Collins",
    "Xavier Stewart",
    "Yara Sanchez",
    "Zoe Morris",
]

TASK_INSTRUCTION = f"""The workspace contains attendance and payroll reference files in `{ATTENDANCE_DIR.as_posix()}`.

Use `april-attendance-data.csv`, `employee-department.txt`, and `salary-rates.txt` to create `{OUTPUT_FILENAME}` in the workspace root.

For each employee:
- Average work length is the mean daily duration in hours across all rows for that employee.
- On-time departure means the clock-out time is between 17:30 and 18:00 inclusive.
- Late arrival means the clock-in time is later than 09:00.
- Total earnings are computed by summing the employee's work hours across all days, rounding that total up to the next integer, and then multiplying by the employee's hourly rate. Do not round individual days before summing.

Then aggregate the employee-level results by department. The output workbook must contain exactly one sheet with columns:
- `Department`
- `Department Average Work Length`
- `Department Average On-time Departure Count`
- `Department Average Late Arrival Count`
- `Department Total Payroll`
"""


def build_prompt() -> str:
    return TASK_INSTRUCTION


def build_example(index: int, seed: int) -> dict:
    rng = random.Random(seed)
    num_departments = rng.randint(3, 5)
    min_employees = num_departments * 2
    max_employees = min(len(NAME_POOL), num_departments * 3)
    num_employees = rng.randint(min_employees, max_employees)
    num_days = rng.randint(10, 18)
    departments = rng.sample(DEPARTMENT_POOL, num_departments)
    employees = rng.sample(NAME_POOL, num_employees)
    assignments = _build_department_assignments(rng, departments, employees)
    rates = _build_rates(rng, employees)
    attendance_rows = _build_attendance_rows(rng, employees, num_days)
    return {
        "id": f"attendance_payroll_audit_{index}",
        "seed": seed,
        "num_departments": num_departments,
        "num_employees": num_employees,
        "num_days": num_days,
        "departments": departments,
        "employees": sorted(employees),
        "department_assignments": assignments,
        "rates": rates,
        "attendance_rows": attendance_rows,
        "prompt": build_prompt(),
    }


def _time_string(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def _build_department_assignments(
    rng: random.Random,
    departments: list[str],
    employees: list[str],
) -> list[dict]:
    assignments = []
    shuffled_employees = employees[:]
    rng.shuffle(shuffled_employees)

    for index, name in enumerate(shuffled_employees):
        if index < len(departments):
            department = departments[index]
        else:
            department = rng.choice(departments)
        assignments.append({"Name": name, "Department": department})

    assignments.sort(key=lambda row: row["Name"])
    return assignments


def _build_rates(
    rng: random.Random,
    employees: list[str],
) -> list[dict]:
    rows = []
    for name in sorted(employees):
        whole = rng.randint(22, 58)
        fractional = rng.choice([0.0, 0.25, 0.5, 0.75])
        rows.append({"Name": name, "Rate": round(whole + fractional, 2)})
    return rows


def _build_attendance_rows(
    rng: random.Random,
    employees: list[str],
    num_days: int,
) -> list[dict]:
    clock_in_options = [
        (8, 35),
        (8, 45),
        (8, 55),
        (9, 0),
        (9, 5),
        (9, 15),
        (9, 25),
    ]
    clock_out_options = [
        (17, 20),
        (17, 30),
        (17, 40),
        (17, 50),
        (18, 0),
        (18, 10),
    ]

    rows = []
    for day in range(num_days):
        date_string = f"2024-04-{day + 1:02d}"
        for name in sorted(employees):
            clock_in = rng.choice(clock_in_options)
            clock_out = rng.choice(clock_out_options)
            rows.append(
                {
                    "Date": date_string,
                    "Name": name,
                    "Clock-in": _time_string(*clock_in),
                    "Clock-out": _time_string(*clock_out),
                }
            )
    return rows


def _format_department_lines(assignments_df: pd.DataFrame) -> str:
    lines = []
    grouped = assignments_df.groupby("Department")["Name"].apply(list).reset_index()
    for row in grouped.sort_values("Department").itertuples(index=False):
        lines.append(f"• {row.Department}: {', '.join(row.Name)}")
    return "\n".join(lines)


def _analyze_attendance(attendance_df: pd.DataFrame) -> pd.DataFrame:
    df = attendance_df.copy()
    df["Clock-in"] = pd.to_datetime(df["Clock-in"], format="%H:%M")
    df["Clock-out"] = pd.to_datetime(df["Clock-out"], format="%H:%M")
    df["Work Minutes"] = (df["Clock-out"] - df["Clock-in"]).dt.total_seconds() / 60.0
    df["Work Length"] = df["Work Minutes"] / 60.0
    df["On-Time Departure"] = df["Clock-out"].dt.strftime("%H:%M").between("17:30", "18:00")
    df["Late Arrival"] = df["Clock-in"].dt.strftime("%H:%M") > "09:00"

    employee_df = (
        df.groupby("Name")
        .agg(
            Average_Work_Length=("Work Length", "mean"),
            On_Time_Departure_Count=("On-Time Departure", "sum"),
            Late_Arrival_Count=("Late Arrival", "sum"),
            Total_Work_Minutes=("Work Minutes", "sum"),
        )
        .reset_index()
    )
    employee_df["Rounded_Total_Work_Length"] = (
        employee_df["Total_Work_Minutes"] / 60.0
    ).apply(math.ceil)
    return employee_df


def build_expected_output(
    attendance_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    rates_df: pd.DataFrame,
) -> pd.DataFrame:
    employee_df = _analyze_attendance(attendance_df)
    employee_df = employee_df.merge(assignments_df, on="Name", how="left")
    employee_df = employee_df.merge(rates_df, on="Name", how="left")
    employee_df["Total Payroll"] = (
        employee_df["Rounded_Total_Work_Length"] * employee_df["Rate"]
    ).round(2)

    department_df = (
        employee_df.groupby("Department")
        .agg(
            Department_Average_Work_Length=("Average_Work_Length", "mean"),
            Department_Average_On_Time_Departure_Count=("On_Time_Departure_Count", "mean"),
            Department_Average_Late_Arrival_Count=("Late_Arrival_Count", "mean"),
            Department_Total_Payroll=("Total Payroll", "sum"),
        )
        .reset_index()
    )
    department_df["Department_Total_Payroll"] = department_df["Department_Total_Payroll"].round(2)
    department_df = department_df.rename(
        columns={
            "Department_Average_Work_Length": "Department Average Work Length",
            "Department_Average_On_Time_Departure_Count": "Department Average On-time Departure Count",
            "Department_Average_Late_Arrival_Count": "Department Average Late Arrival Count",
            "Department_Total_Payroll": "Department Total Payroll",
        }
    )
    return department_df.sort_values("Department").reset_index(drop=True)


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    workspace = Path(workspace_dir)
    attendance_dir = workspace / ATTENDANCE_DIR
    attendance_dir.mkdir(parents=True, exist_ok=True)

    assignments_df = pd.DataFrame(example["department_assignments"])
    rates_df = pd.DataFrame(example["rates"])
    attendance_df = pd.DataFrame(example["attendance_rows"])
    expected_df = build_expected_output(attendance_df, assignments_df, rates_df)

    attendance_df.to_csv(workspace / ATTENDANCE_DOC_PATH, index=False)
    (workspace / DEPARTMENT_DOC_PATH).write_text(
        _format_department_lines(assignments_df),
        encoding="utf-8",
    )
    (workspace / RATES_DOC_PATH).write_text(
        "\n".join(
            f"{row.Name}: {row.Rate:.2f}"
            for row in rates_df.sort_values("Name").itertuples(index=False)
        ),
        encoding="utf-8",
    )

    groundtruth_dir = Path(log_dir) / "groundtruth"
    groundtruth_dir.mkdir(parents=True, exist_ok=True)
    expected_df.to_csv(groundtruth_dir / "expected.csv", index=False)
    (groundtruth_dir / "example.json").write_text(
        json.dumps(example, indent=2),
        encoding="utf-8",
    )
