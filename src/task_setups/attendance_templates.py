"""Declarative template specs for the attendance-payroll-audit diversity sweep.

Each template composes:

- docs:    document atoms to materialize into the workspace
- compute: which computation atom produces the expected aggregate table
- actions: which deliverable checkpoints (primary table + optional artifacts)
- goal_prompt: the goal-level instruction shown to the agent. Prompts state
  the required policy and deliverable but do not enumerate the algorithm
  step-by-step.

Diversity variants are named slices of the registry.
"""

from __future__ import annotations


_DISCOVER = (
    "Inspect the reference files in the attendance directory before deciding "
    "how to aggregate; the schema and policies are described there. Write all "
    "requested output files in the workspace root, which is the directory that "
    "contains the `Documents` folder. Report average columns to four decimal "
    "places and payroll or bonus columns to two decimal places."
)


def _prompt(*paragraphs: str) -> str:
    return "\n\n".join(list(paragraphs) + [_DISCOVER])


_BASIC_COLUMNS_TEXT = (
    "Output columns: `Department`, `Department Average Work Length`, "
    "`Department Average On-time Departure Count`, "
    "`Department Average Late Arrival Count`, `Department Total Payroll`."
)


_TEAM_COLUMNS_TEXT = (
    "Output columns: `Team`, `Team Average Work Length`, "
    "`Team Average On-time Departure Count`, "
    "`Team Average Late Arrival Count`, `Team Total Payroll`."
)


_ROLE_COLUMNS_TEXT = (
    "Output columns: `Role`, `Role Average Work Length`, "
    "`Role Average On-time Departure Count`, "
    "`Role Average Late Arrival Count`, `Role Total Payroll`."
)


_BONUS_COLUMNS_TEXT = (
    "Output columns: `Department`, `Department Average Work Length`, "
    "`Department Average On-time Departure Count`, "
    "`Department Average Late Arrival Count`, `Department Total Payroll`, "
    "`Department Total Bonus`."
)


_NET_COLUMNS_TEXT = (
    "Output columns: `Department`, `Department Average Work Length`, "
    "`Department Average On-time Departure Count`, "
    "`Department Average Late Arrival Count`, `Department Total Payroll`, "
    "`Department Net Payroll`."
)


_STANDARD_RULES = (
    "Average work length is the mean daily duration in hours per employee. "
    "On-time departure means the clock-out time is between 17:30 and 18:00 "
    "inclusive. Late arrival means the clock-in time is later than 09:00. "
    "Total payroll is each employee's total work hours summed across days, "
    "rounded up to the next integer, multiplied by the employee's hourly "
    "rate (no rounding of daily hours before summing)."
)


_STANDARD_ATTENDANCE_RULES = (
    "Average work length is the mean daily duration in hours per employee. "
    "On-time departure means the clock-out time is between 17:30 and 18:00 "
    "inclusive. Late arrival means the clock-in time is later than 09:00."
)


TEMPLATES: dict[str, dict] = {
    "T01_dept_basic_xlsx": {
        "id": "T01_dept_basic_xlsx",
        "docs": ["attendance", "departments", "rates"],
        "compute": "basic_dept",
        "actions": ["write_xlsx_basic"],
        "goal_prompt": _prompt(
            "Produce a department-level attendance and payroll audit. "
            f"{_STANDARD_RULES}",
            "Aggregate the employee-level metrics into a department report "
            "saved as `attendance-payroll-audit.xlsx` in the workspace root.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T02_dept_basic_csv": {
        "id": "T02_dept_basic_csv",
        "docs": ["attendance", "departments", "rates"],
        "compute": "basic_dept",
        "actions": ["write_csv_basic"],
        "goal_prompt": _prompt(
            "Produce a department-level attendance and payroll audit. "
            f"{_STANDARD_RULES}",
            "Save the department-aggregated report as "
            "`attendance-payroll-audit.csv` in the workspace root.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T03_dept_basic_json": {
        "id": "T03_dept_basic_json",
        "docs": ["attendance", "departments", "rates"],
        "compute": "basic_dept",
        "actions": ["write_json_basic"],
        "goal_prompt": _prompt(
            "Produce a department-level attendance and payroll audit. "
            f"{_STANDARD_RULES}",
            "Save the department-aggregated report as "
            "`attendance-payroll-audit.json` in the workspace root. The JSON "
            "must be a list of records (one per department).",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T04_dept_overtime_xlsx": {
        "id": "T04_dept_overtime_xlsx",
        "docs": ["attendance", "departments", "rates", "overtime_policy"],
        "compute": "overtime_dept",
        "actions": ["write_xlsx_basic"],
        "goal_prompt": _prompt(
            "Produce a department-level audit that applies the company's "
            "overtime policy when computing payroll. The overtime-policy "
            "document defines how hours above eight per day are paid. "
            f"{_STANDARD_ATTENDANCE_RULES}",
            "Save the result as `attendance-payroll-audit.xlsx` in the "
            "workspace root.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T05_dept_holiday_xlsx": {
        "id": "T05_dept_holiday_xlsx",
        "docs": ["attendance", "departments", "rates", "holidays"],
        "compute": "holiday_adj_dept",
        "actions": ["write_xlsx_basic"],
        "goal_prompt": _prompt(
            "Produce a department-level audit. Some dates in the period are "
            "company holidays (see holidays document). Attendance rows on "
            "those dates must not count toward on-time-departure or "
            "late-arrival aggregates. For non-holiday attendance rows, "
            "on-time departure means the clock-out time is between 17:30 and "
            "18:00 inclusive, and late arrival means the clock-in time is "
            "later than 09:00. Average work length is the mean daily duration "
            "in hours per employee and still includes all rows. Total payroll "
            "also includes all rows and is each employee's total work hours "
            "summed across days, rounded up to the next integer, multiplied "
            "by the employee's hourly rate.",
            "Save the result as `attendance-payroll-audit.xlsx` in the "
            "workspace root.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T06_dept_probation_xlsx": {
        "id": "T06_dept_probation_xlsx",
        "docs": ["attendance", "departments", "rates", "probation"],
        "compute": "probation_dept",
        "actions": ["write_xlsx_basic"],
        "goal_prompt": _prompt(
            "Produce a department-level audit. The probation document lists "
            "employees currently on probation; exclude them entirely from "
            "the department-level aggregates and from total payroll. "
            f"For included employees, {_STANDARD_RULES.lower()}",
            "Save the result as `attendance-payroll-audit.xlsx` in the "
            "workspace root.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T07_dept_shift_xlsx": {
        "id": "T07_dept_shift_xlsx",
        "docs": ["attendance", "departments", "rates", "shift_schedule"],
        "compute": "shift_dept",
        "actions": ["write_xlsx_basic"],
        "goal_prompt": _prompt(
            "Produce a department-level audit using each employee's shift "
            "schedule. On-time departure means the clock-out is within "
            "15 minutes of the employee's expected clock-out. Late arrival "
            "means clock-in is later than the employee's expected clock-in. "
            "Average work length is the mean daily duration in hours per "
            "employee. Total payroll is each employee's total work hours "
            "summed across days, rounded up to the next integer, multiplied "
            "by the employee's hourly rate.",
            "Save the result as `attendance-payroll-audit.xlsx` in the "
            "workspace root.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T08_dept_bonus_xlsx": {
        "id": "T08_dept_bonus_xlsx",
        "docs": ["attendance", "departments", "rates", "bonus_policy"],
        "compute": "bonus_dept",
        "actions": ["write_xlsx_with_bonus"],
        "goal_prompt": _prompt(
            "Produce a department-level audit that includes a bonus column. "
            f"{_STANDARD_RULES} The bonus policy in the workspace describes "
            "who qualifies and at what rate; sum each qualifying employee's "
            "bonus into a `Department Total Bonus` column.",
            "Save the result as `attendance-payroll-audit.xlsx` in the "
            "workspace root.",
            _BONUS_COLUMNS_TEXT,
        ),
    },
    "T09_dept_tax_xlsx": {
        "id": "T09_dept_tax_xlsx",
        "docs": ["attendance", "departments", "rates", "tax_table"],
        "compute": "tax_dept",
        "actions": ["write_xlsx_net"],
        "goal_prompt": _prompt(
            "Produce a department-level audit that reports both gross and "
            f"net payroll. {_STANDARD_RULES} Apply the tax table to each "
            "employee's gross payroll by selecting the first bracket whose "
            "maximum payroll is greater than or equal to that employee's "
            "gross payroll. The selected tax rate is a flat rate applied to "
            "the employee's full gross payroll, not a marginal bracket "
            "calculation. Sum employee net payroll into a `Department Net "
            "Payroll` column.",
            "Save the result as `attendance-payroll-audit.xlsx` in the "
            "workspace root.",
            _NET_COLUMNS_TEXT,
        ),
    },
    "T10_team_xlsx": {
        "id": "T10_team_xlsx",
        "docs": ["attendance", "departments", "rates", "teams"],
        "compute": "team_agg",
        "actions": ["write_xlsx_team"],
        "goal_prompt": _prompt(
            "Produce a team-level (not department-level) attendance and "
            "payroll audit using the team mapping document. "
            f"{_STANDARD_RULES}",
            "Save the team report as `team-payroll-audit.xlsx` in the "
            "workspace root.",
            _TEAM_COLUMNS_TEXT,
        ),
    },
    "T11_role_xlsx": {
        "id": "T11_role_xlsx",
        "docs": ["attendance", "departments", "rates", "roles"],
        "compute": "role_agg",
        "actions": ["write_xlsx_role"],
        "goal_prompt": _prompt(
            "Produce a role-level (not department-level) attendance and "
            "payroll audit using the role mapping document. "
            f"{_STANDARD_RULES}",
            "Save the role report as `role-payroll-audit.xlsx` in the "
            "workspace root.",
            _ROLE_COLUMNS_TEXT,
        ),
    },
    "T12_dept_with_exceptions": {
        "id": "T12_dept_with_exceptions",
        "docs": ["attendance", "departments", "rates"],
        "compute": "basic_dept",
        "actions": ["write_xlsx_basic", "write_exceptions"],
        "goal_prompt": _prompt(
            "Produce a department-level audit. "
            f"{_STANDARD_RULES}",
            "Save the report as `attendance-payroll-audit.xlsx` in the "
            "workspace root. In addition, write `exceptions.txt` listing "
            "every employee whose late-arrival count is 3 or more (one "
            "employee per line).",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T13_dept_with_summary": {
        "id": "T13_dept_with_summary",
        "docs": ["attendance", "departments", "rates"],
        "compute": "basic_dept",
        "actions": ["write_xlsx_basic", "write_summary"],
        "goal_prompt": _prompt(
            "Produce a department-level audit. "
            f"{_STANDARD_RULES}",
            "Save the report as `attendance-payroll-audit.xlsx` in the "
            "workspace root. Additionally write a narrative `summary.md` in "
            "the workspace root that names every department covered in the "
            "audit.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T14_dept_with_top_late": {
        "id": "T14_dept_with_top_late",
        "docs": ["attendance", "departments", "rates"],
        "compute": "basic_dept",
        "actions": ["write_xlsx_basic", "write_top_late"],
        "goal_prompt": _prompt(
            "Produce a department-level audit. "
            f"{_STANDARD_RULES}",
            "Save the report as `attendance-payroll-audit.xlsx` in the "
            "workspace root. Additionally write `top_late.txt` listing the "
            "three employees with the highest late-arrival counts, ordered "
            "from highest to lowest (one employee per line).",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T15_dept_overtime_holiday_xlsx": {
        "id": "T15_dept_overtime_holiday_xlsx",
        "docs": [
            "attendance", "departments", "rates", "overtime_policy", "holidays",
        ],
        "compute": "overtime_holiday_dept",
        "actions": ["write_xlsx_basic"],
        "goal_prompt": _prompt(
            "Produce a department-level audit that applies the overtime "
            "policy when computing payroll and skips holiday rows when "
            "counting on-time and late events. For non-holiday attendance "
            "rows, on-time departure means the clock-out time is between "
            "17:30 and 18:00 inclusive, and late arrival means the clock-in "
            "time is later than 09:00. Average work length is the mean daily "
            "duration in hours per employee and still includes all attendance "
            "rows.",
            "Save the result as `attendance-payroll-audit.xlsx` in the "
            "workspace root.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T16_dept_probation_overtime_xlsx": {
        "id": "T16_dept_probation_overtime_xlsx",
        "docs": [
            "attendance", "departments", "rates", "probation", "overtime_policy",
        ],
        "compute": "probation_overtime_dept",
        "actions": ["write_xlsx_basic"],
        "goal_prompt": _prompt(
            "Produce a department-level audit. Exclude employees listed in "
            "the probation document. For included employees, apply the "
            "overtime policy when computing payroll. "
            f"{_STANDARD_ATTENDANCE_RULES}",
            "Save the result as `attendance-payroll-audit.xlsx` in the "
            "workspace root.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T17_dept_minwage_xlsx": {
        "id": "T17_dept_minwage_xlsx",
        "docs": ["attendance", "departments", "rates", "minimum_wage"],
        "compute": "basic_dept",
        "actions": ["write_xlsx_basic", "write_violations"],
        "goal_prompt": _prompt(
            "Produce a department-level audit. "
            f"{_STANDARD_RULES}",
            "Save the report as `attendance-payroll-audit.xlsx` in the "
            "workspace root. Additionally write `violations.txt` listing "
            "every employee whose hourly rate is below the minimum wage "
            "defined in the minimum-wage document (one employee per line).",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T18_dept_artifacts_only": {
        "id": "T18_dept_artifacts_only",
        "docs": ["attendance", "departments", "rates"],
        "compute": "basic_dept",
        "actions": ["write_summary", "write_exceptions"],
        "goal_prompt": _prompt(
            "Audit attendance and payroll at the department level, but the "
            "only deliverables are two text artifacts. "
            f"{_STANDARD_RULES}",
            "Write `summary.md` naming every department covered. Write "
            "`exceptions.txt` listing every employee with three or more late "
            "arrivals.",
        ),
    },
    "T19_dept_csv_summary": {
        "id": "T19_dept_csv_summary",
        "docs": ["attendance", "departments", "rates"],
        "compute": "basic_dept",
        "actions": ["write_csv_basic", "write_summary"],
        "goal_prompt": _prompt(
            "Produce a department-level audit. "
            f"{_STANDARD_RULES}",
            "Save the report as `attendance-payroll-audit.csv` in the "
            "workspace root. Additionally write `summary.md` naming every "
            "department covered.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
    "T20_dept_full_audit": {
        "id": "T20_dept_full_audit",
        "docs": [
            "attendance", "departments", "rates",
            "overtime_policy", "holidays", "minimum_wage",
        ],
        "compute": "overtime_holiday_dept",
        "actions": [
            "write_xlsx_basic",
            "write_summary",
            "write_exceptions",
            "write_violations",
        ],
        "goal_prompt": _prompt(
            "Produce a full department-level audit applying all current "
            "policies. Apply the overtime policy when computing payroll, "
            "skip holiday rows when counting on-time and late events, and "
            "produce the department aggregates. For non-holiday attendance "
            "rows, on-time departure means the clock-out time is between "
            "17:30 and 18:00 inclusive, and late arrival means the clock-in "
            "time is later than 09:00. Average work length is the mean daily "
            "duration in hours per employee and still includes all attendance "
            "rows.",
            "Save the primary audit as `attendance-payroll-audit.xlsx`. "
            "Additionally write `summary.md` naming every department, "
            "`exceptions.txt` listing employees with 3+ late arrivals, and "
            "`violations.txt` listing employees whose hourly rate is below "
            "the minimum-wage threshold.",
            _BASIC_COLUMNS_TEXT,
        ),
    },
}


# Diversity variants — named slices of the registry.
# Counts: low=1, medium=3, high=8, extra_high=20
VARIANTS: dict[str, list[str]] = {
    "low": ["T01_dept_basic_xlsx"],
    "medium": [
        "T01_dept_basic_xlsx",
        "T02_dept_basic_csv",
        "T04_dept_overtime_xlsx",
    ],
    "high": [
        "T01_dept_basic_xlsx",
        "T02_dept_basic_csv",
        "T03_dept_basic_json",
        "T04_dept_overtime_xlsx",
        "T05_dept_holiday_xlsx",
        "T06_dept_probation_xlsx",
        "T10_team_xlsx",
        "T12_dept_with_exceptions",
    ],
    "extra_high": list(TEMPLATES.keys()),
}
