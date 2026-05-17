"""Template-driven setup for the attendance-payroll-audit task family.

The example dict carries a `template_id` field selecting one of the templates
in `attendance_templates.TEMPLATES`. The template spec drives:

- which reference documents are materialized into the workspace
- which computation produces the `expected.json` ground truth
- which deliverable artifacts the eval will check

All variants (`low` / `medium` / `high` / `extra_high`) share this module;
the only thing that changes between variants is which subset of templates
appears in the generated data file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from .attendance_atoms import (
    ATTENDANCE_DIR,
    DOC_BUILDERS,
    DOC_PATHS,
    build_expected,
)
from .attendance_templates import TEMPLATES


# Re-export commonly used paths so callers (eval, generation scripts) can
# stay agnostic of the atoms-module layout.
ATTENDANCE_DOC_PATH = DOC_PATHS["attendance"]
DEPARTMENT_DOC_PATH = DOC_PATHS["departments"]
RATES_DOC_PATH = DOC_PATHS["rates"]
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
    "Ava Thompson", "Benjamin Lee", "Charlotte Davis", "Daniel Walker",
    "Ella Martin", "Felix Young", "Grace Hall", "Henry Allen",
    "Isla Scott", "Jack Green", "Katherine Baker", "Liam Adams",
    "Mia Nelson", "Noah Carter", "Olivia Perez", "Patrick Roberts",
    "Quinn Turner", "Ruby Phillips", "Samuel Campbell", "Tessa Parker",
    "Uma Evans", "Victor Edwards", "Willow Collins", "Xavier Stewart",
    "Yara Sanchez", "Zoe Morris",
]


TEAM_POOL = [
    "Acquisition", "Retention", "Platform", "Research", "Field Ops",
    "Customer Success", "Brand", "Analytics",
]


ROLE_POOL = ["Manager", "Senior", "Engineer", "Analyst", "Specialist"]


CLOCK_IN_OPTIONS = [
    (8, 35), (8, 45), (8, 55), (9, 0), (9, 5), (9, 15), (9, 25),
]


CLOCK_OUT_OPTIONS = [
    (17, 20), (17, 30), (17, 40), (17, 50), (18, 0), (18, 10), (18, 30), (19, 0),
]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _template_for(example: dict) -> dict:
    template_id = example.get("template_id")
    if template_id is None:
        # Backwards compat: pre-template data files used the original prompt;
        # map them onto the basic template.
        template_id = "T01_dept_basic_xlsx"
    return TEMPLATES[template_id]


def build_prompt(example: dict | None = None) -> str:
    if example is None:
        return TEMPLATES["T01_dept_basic_xlsx"]["goal_prompt"]
    return _template_for(example)["goal_prompt"]


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------


def _time_string(hour: int, minute: int) -> str:
    return f"{hour:02d}:{minute:02d}"


def _base_skeleton(rng: random.Random) -> dict:
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
        "num_departments": num_departments,
        "num_employees": num_employees,
        "num_days": num_days,
        "departments": departments,
        "employees": sorted(employees),
        "department_assignments": assignments,
        "rates": rates,
        "attendance_rows": attendance_rows,
    }


def _build_department_assignments(rng, departments, employees):
    assignments = []
    shuffled = employees[:]
    rng.shuffle(shuffled)
    for idx, name in enumerate(shuffled):
        if idx < len(departments):
            department = departments[idx]
        else:
            department = rng.choice(departments)
        assignments.append({"Name": name, "Department": department})
    assignments.sort(key=lambda row: row["Name"])
    return assignments


def _build_rates(rng, employees):
    rows = []
    for name in sorted(employees):
        whole = rng.randint(22, 58)
        fractional = rng.choice([0.0, 0.25, 0.5, 0.75])
        rows.append({"Name": name, "Rate": round(whole + fractional, 2)})
    return rows


def _build_attendance_rows(rng, employees, num_days):
    rows = []
    for day in range(num_days):
        date_string = f"2024-04-{day + 1:02d}"
        for name in sorted(employees):
            clock_in = rng.choice(CLOCK_IN_OPTIONS)
            clock_out = rng.choice(CLOCK_OUT_OPTIONS)
            rows.append(
                {
                    "Date": date_string,
                    "Name": name,
                    "Clock-in": _time_string(*clock_in),
                    "Clock-out": _time_string(*clock_out),
                }
            )
    return rows


def _attach_holidays(example: dict, rng: random.Random) -> None:
    num_holidays = rng.randint(2, max(2, example["num_days"] // 4))
    dates = sorted({f"2024-04-{rng.randint(1, example['num_days']):02d}" for _ in range(num_holidays * 2)})[:num_holidays]
    names = ["Spring Festival", "Founders Day", "Memorial Pause", "Quarterly Closeout", "Wellness Day"]
    holidays = []
    for i, date in enumerate(dates):
        holidays.append({"Date": date, "Name": names[i % len(names)]})
    example["holidays"] = holidays


def _attach_probation(example: dict, rng: random.Random) -> None:
    n = max(1, len(example["employees"]) // 5)
    example["probation"] = sorted(rng.sample(example["employees"], n))


def _attach_shifts(example: dict, rng: random.Random) -> None:
    shifts = []
    for name in example["employees"]:
        in_h = rng.choice([8, 9, 10])
        in_m = rng.choice([0, 30])
        out_h = in_h + 9
        out_m = in_m
        shifts.append(
            {
                "Name": name,
                "Expected_Clock_In": _time_string(in_h, in_m),
                "Expected_Clock_Out": _time_string(out_h % 24, out_m),
            }
        )
    example["shifts"] = shifts


def _attach_teams(example: dict, rng: random.Random) -> None:
    num_teams = min(rng.randint(2, 4), len(example["employees"]))
    teams = rng.sample(TEAM_POOL, num_teams)
    assignments = []
    shuffled = example["employees"][:]
    rng.shuffle(shuffled)
    for idx, name in enumerate(shuffled):
        team = teams[idx] if idx < num_teams else rng.choice(teams)
        assignments.append({"Name": name, "Team": team})
    assignments.sort(key=lambda r: r["Name"])
    example["team_assignments"] = assignments


def _attach_roles(example: dict, rng: random.Random) -> None:
    assignments = []
    for idx, name in enumerate(sorted(example["employees"])):
        role = ROLE_POOL[idx % len(ROLE_POOL)]
        assignments.append({"Name": name, "Role": role})
    rng.shuffle(assignments)
    assignments.sort(key=lambda r: r["Name"])
    example["role_assignments"] = assignments


def _attach_tax_table(example: dict, rng: random.Random) -> None:
    example["tax_table"] = [
        {"Max": 4000.0, "Rate": 0.10},
        {"Max": 9000.0, "Rate": 0.18},
        {"Max": None, "Rate": 0.25},
    ]


def _attach_bonus_policy(example: dict, rng: random.Random) -> None:
    example["bonus_policy"] = {
        "on_time_threshold": rng.randint(4, 7),
        "bonus_pct": rng.choice([0.03, 0.05, 0.07]),
    }


def _attach_minimum_wage(example: dict, rng: random.Random) -> None:
    example["minimum_wage_threshold"] = float(rng.choice([28.0, 30.0, 32.0, 34.0]))


_ATTACH_FOR_DOC = {
    "holidays": _attach_holidays,
    "probation": _attach_probation,
    "shift_schedule": _attach_shifts,
    "teams": _attach_teams,
    "roles": _attach_roles,
    "tax_table": _attach_tax_table,
    "bonus_policy": _attach_bonus_policy,
    "minimum_wage": _attach_minimum_wage,
}


def build_example(index: int, seed: int, template_id: str = "T01_dept_basic_xlsx") -> dict:
    rng = random.Random(seed)
    template = TEMPLATES[template_id]

    base = _base_skeleton(rng)
    example: dict = {
        "id": f"attendance_payroll_audit_{index}",
        "seed": seed,
        "template_id": template_id,
        **base,
    }
    for doc in template["docs"]:
        attach = _ATTACH_FOR_DOC.get(doc)
        if attach is not None:
            attach(example, rng)

    example["prompt"] = build_prompt(example)
    return example


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    template = _template_for(example)
    workspace = Path(workspace_dir)
    attendance_dir = workspace / ATTENDANCE_DIR
    attendance_dir.mkdir(parents=True, exist_ok=True)

    for doc in template["docs"]:
        path = workspace / DOC_PATHS[doc]
        path.parent.mkdir(parents=True, exist_ok=True)
        content = DOC_BUILDERS[doc](example)
        path.write_text(content, encoding="utf-8")

    expected = build_expected(example, template)
    groundtruth_dir = Path(log_dir) / "groundtruth"
    groundtruth_dir.mkdir(parents=True, exist_ok=True)
    (groundtruth_dir / "expected.json").write_text(
        json.dumps(expected, indent=2, default=str),
        encoding="utf-8",
    )
    (groundtruth_dir / "example.json").write_text(
        json.dumps(example, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Legacy export retained for compatibility with older eval code paths
# ---------------------------------------------------------------------------


def build_expected_output(
    attendance_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
    rates_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct the original basic-template expected DataFrame.

    Older data files (no template_id) and existing direct callers use this
    helper to produce the department-level audit table for the basic task.
    """
    example = {
        "attendance_rows": attendance_df.to_dict(orient="records"),
        "department_assignments": assignments_df.to_dict(orient="records"),
        "rates": rates_df.to_dict(orient="records"),
    }
    from .attendance_atoms import compute_basic_dept
    return compute_basic_dept(example)["primary_df"]
