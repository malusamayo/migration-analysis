"""Declarative template specs for the budget-approval diversity sweep.

A template composes:

- queries: extra contact roles to populate (`finance`, `hr`, `cfo`, `vendor`)
- docs:    document atoms to materialize into the mock filesystem
- compute: which computation atom drives the expected outcomes
- actions: which action atoms the agent must satisfy (each becomes one
           checkpoint in the eval)
- goal_prompt: the goal-level instruction shown to the agent. Prompts are
  deliberately under-specified — they state the goal and constraints but do
  *not* enumerate steps, name documents, or name contacts. The agent is
  expected to discover both via the Agent Company tools.

Diversity variants are simply named slices of this registry.
"""

from __future__ import annotations

# Goal-level prompts share a common suffix to keep style uniform across templates.
_DISCOVER = (
    "Discover the available contacts and documents through the Agent Company tools "
    "before deciding how to proceed."
)


def _prompt(*paragraphs: str) -> str:
    return "\n\n".join(list(paragraphs) + [_DISCOVER])


TEMPLATES: dict[str, dict] = {
    "T01_dept_total_reply": {
        "id": "T01_dept_total_reply",
        "queries": [],
        "docs": ["products", "dept_budgets"],
        "compute": "per_department",
        "actions": ["reply_total_status"],
        "goal_prompt": _prompt(
            "Your task is to review this period's equipment requests against the company's "
            "department budgets. For each requesting manager, send them a reply stating the "
            "total cost of their request and whether their department's combined requests "
            "stay within its budget."
        ),
    },
    "T02_personal_remaining_reply": {
        "id": "T02_personal_remaining_reply",
        "queries": ["finance"],
        "docs": ["products"],
        "compute": "remaining_personal",
        "actions": ["reply_total_status"],
        "goal_prompt": _prompt(
            "Each employee has a personal budget for this period that has been partially "
            "used. For every employee with an equipment request, send them a reply stating "
            "the total cost of their request and whether it fits within their remaining "
            "personal allowance."
        ),
    },
    "T03_dept_reduction": {
        "id": "T03_dept_reduction",
        "queries": [],
        "docs": ["products", "dept_budgets", "reduction_policy"],
        "compute": "per_department",
        "actions": ["reply_total_only", "reduction_requests", "reduction_record"],
        "goal_prompt": _prompt(
            "Process this period's equipment requests so that every department ends up "
            "within its budget. Reply to each requesting manager with the total cost of "
            "their request. For any over-budget department, apply the company's reduction "
            "policy to bring the department within budget and record the confirmed "
            "equipment removals in `result.txt` in your workspace."
        ),
    },
    "T04_dept_volume_discount_reply": {
        "id": "T04_dept_volume_discount_reply",
        "queries": [],
        "docs": ["products", "dept_budgets", "vendor_terms"],
        "compute": "with_volume_discount",
        "actions": ["reply_total_status"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets. The company has volume-discount terms with its equipment vendor that "
            "affect the effective cost. For each requesting manager, send them a reply "
            "stating their total request cost and whether their department's combined "
            "requests stay within its budget after any applicable discount."
        ),
    },
    "T05_dept_emergency_fund_reply": {
        "id": "T05_dept_emergency_fund_reply",
        "queries": [],
        "docs": ["products", "dept_budgets", "emergency_fund"],
        "compute": "with_emergency_fund",
        "actions": ["reply_total_status"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets, including any emergency reserve each department has access to. For "
            "each requesting manager, send them a reply stating their total request cost "
            "and whether their department's combined requests fit within its budget "
            "(reserve included)."
        ),
    },
    "T06_dept_restricted_categories_reply": {
        "id": "T06_dept_restricted_categories_reply",
        "queries": [],
        "docs": ["products", "dept_budgets", "categories"],
        "compute": "restricted_filtered",
        "actions": ["reply_total_status"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets. Restricted-category items do not count toward department budgets. For "
            "each requesting manager, send them a reply stating their effective total "
            "(excluding restricted items) and whether their department's combined requests "
            "fit within its budget."
        ),
    },
    "T07_dept_headcount_allocated_reply": {
        "id": "T07_dept_headcount_allocated_reply",
        "queries": [],
        "docs": ["products", "budget_pool", "headcount"],
        "compute": "headcount_allocated",
        "actions": ["reply_total_status"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's budget pool, "
            "which is allocated across departments by headcount share. For each requesting "
            "manager, send them a reply stating their total request cost and whether their "
            "department's combined requests fit within its allocated share."
        ),
    },
    "T08_dept_over_cap_escalation": {
        "id": "T08_dept_over_cap_escalation",
        "queries": ["cfo"],
        "docs": ["products", "dept_budgets", "approval_policy"],
        "compute": "over_cap_items",
        "actions": ["reply_total_status", "escalate_to_cfo"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets. Any single line item priced above the company's approval cap must be "
            "escalated to the CFO. Reply to each requesting manager with their total "
            "request cost and budget status, and separately notify the CFO of every "
            "over-cap line item."
        ),
    },
    "T09_dept_total_write_approval": {
        "id": "T09_dept_total_write_approval",
        "queries": [],
        "docs": ["products", "dept_budgets"],
        "compute": "per_department",
        "actions": ["write_approval_txt"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets. Write a file `approval.txt` in your workspace with one approval line "
            "per requesting manager, in the form "
            "`ManagerName: under budget @ $TOTAL` or "
            "`ManagerName: exceed budget @ $TOTAL`. No replies to managers are needed."
        ),
    },
    "T10_dept_total_write_summary": {
        "id": "T10_dept_total_write_summary",
        "queries": [],
        "docs": ["products", "dept_budgets"],
        "compute": "per_department",
        "actions": ["reply_total_status", "write_summary_txt"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets. Reply to each manager with their request total and budget status, "
            "and additionally write a department-level summary to `summary.txt` in your "
            "workspace listing each department's combined request total."
        ),
    },
    "T11_dept_headcount_reduction": {
        "id": "T11_dept_headcount_reduction",
        "queries": [],
        "docs": ["products", "budget_pool", "headcount", "reduction_policy"],
        "compute": "headcount_allocated",
        "actions": ["reply_total_only", "reduction_requests", "reduction_record"],
        "goal_prompt": _prompt(
            "Process this period's equipment requests so that every department ends up "
            "within its headcount-allocated share of the budget pool. Reply to each "
            "requesting manager with the total cost of their request. For any over-"
            "allocation department, apply the company's reduction policy to bring the "
            "department within its share and record the confirmed equipment removals in "
            "`result.txt` in your workspace."
        ),
    },
    "T12_dept_emergency_fund_reduction": {
        "id": "T12_dept_emergency_fund_reduction",
        "queries": [],
        "docs": ["products", "dept_budgets", "emergency_fund", "reduction_policy"],
        "compute": "with_emergency_fund",
        "actions": ["reply_total_only", "reduction_requests", "reduction_record"],
        "goal_prompt": _prompt(
            "Process this period's equipment requests so that every department ends up "
            "within its budget, including the emergency reserve. Reply to each requesting "
            "manager with the total cost of their request. For any department still over "
            "budget after applying its reserve, apply the company's reduction policy to "
            "bring it within budget+reserve and record the confirmed equipment removals in "
            "`result.txt` in your workspace."
        ),
    },
    "T13_dept_restricted_reduction": {
        "id": "T13_dept_restricted_reduction",
        "queries": [],
        "docs": ["products", "dept_budgets", "categories", "reduction_policy"],
        "compute": "restricted_filtered",
        "actions": ["reply_total_only", "reduction_requests", "reduction_record"],
        "goal_prompt": _prompt(
            "Process this period's equipment requests so that every department ends up "
            "within its budget. Restricted-category items do not count toward budget. Reply "
            "to each requesting manager with the effective cost of their (non-restricted) "
            "request, and for any over-budget department apply the company's reduction "
            "policy and record the confirmed equipment removals in `result.txt` in your "
            "workspace."
        ),
    },
    "T14_dept_vendor_consult": {
        "id": "T14_dept_vendor_consult",
        "queries": ["vendor"],
        "docs": ["products", "dept_budgets"],
        "compute": "with_volume_discount",
        "actions": ["reply_total_status"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets. The company has a vendor relationship that provides volume-discount "
            "tiers; consult the vendor representative to learn the current terms. For each "
            "requesting manager, send them a reply stating their total request cost and "
            "whether their department's combined requests fit within its budget after any "
            "applicable discount."
        ),
    },
    "T15_dept_hr_consult_headcount": {
        "id": "T15_dept_hr_consult_headcount",
        "queries": ["hr"],
        "docs": ["products", "budget_pool"],
        "compute": "headcount_allocated",
        "actions": ["reply_total_status"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's budget pool, "
            "which is allocated across departments by headcount share. Headcount is "
            "maintained by HR; consult them for current numbers. For each requesting "
            "manager, send them a reply stating their total request cost and whether their "
            "department's combined requests fit within its allocated share."
        ),
    },
    "T16_dept_cfo_consult_cap": {
        "id": "T16_dept_cfo_consult_cap",
        "queries": ["cfo"],
        "docs": ["products", "dept_budgets"],
        "compute": "over_cap_items",
        "actions": ["reply_total_status", "escalate_to_cfo"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets. The CFO owns the per-line-item approval cap; consult them for the "
            "current value. Reply to each requesting manager with their total request cost "
            "and budget status, and separately notify the CFO of every line item priced "
            "above the cap."
        ),
    },
    "T17_personal_remaining_write_approval": {
        "id": "T17_personal_remaining_write_approval",
        "queries": ["finance"],
        "docs": ["products"],
        "compute": "remaining_personal",
        "actions": ["write_approval_txt"],
        "goal_prompt": _prompt(
            "Each employee has a personal budget for this period that has been partially "
            "used. Write a file `approval.txt` in your workspace with one approval line per "
            "requesting employee, in the form "
            "`EmployeeName: under budget @ $TOTAL` or "
            "`EmployeeName: exceed budget @ $TOTAL`. No replies are needed."
        ),
    },
    "T18_dept_total_artifacts_only": {
        "id": "T18_dept_total_artifacts_only",
        "queries": [],
        "docs": ["products", "dept_budgets"],
        "compute": "per_department",
        "actions": ["write_approval_txt", "write_summary_txt"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets and produce two workspace files: `approval.txt` with one approval line "
            "per requesting manager (in the form "
            "`ManagerName: under budget @ $TOTAL` or "
            "`ManagerName: exceed budget @ $TOTAL`), and `summary.txt` with one line per "
            "department showing the department's combined request total. The files are the "
            "only deliverables; no manager messaging is required."
        ),
    },
    "T19_dept_volume_discount_write_approval": {
        "id": "T19_dept_volume_discount_write_approval",
        "queries": [],
        "docs": ["products", "dept_budgets", "vendor_terms"],
        "compute": "with_volume_discount",
        "actions": ["reply_total_status", "write_approval_txt"],
        "goal_prompt": _prompt(
            "Review this period's equipment requests against the company's department "
            "budgets, accounting for the vendor's current volume-discount terms. Reply to "
            "each requesting manager with their total request cost and discounted budget "
            "status, and additionally write `approval.txt` in your workspace listing one "
            "approval line per requesting manager."
        ),
    },
    "T20_dept_full_pipeline": {
        "id": "T20_dept_full_pipeline",
        "queries": ["cfo"],
        "docs": ["products", "dept_budgets", "approval_policy", "reduction_policy"],
        "compute": "over_cap_items",
        "actions": [
            "reply_total_only",
            "reduction_requests",
            "reduction_record",
            "escalate_to_cfo",
        ],
        "goal_prompt": _prompt(
            "Process this period's equipment requests under all current policies. Any over-"
            "budget department must be reduced using the company's reduction policy, and "
            "any single line item priced above the company's per-item approval cap must be "
            "escalated to the CFO. Reply to each requesting manager with the total cost of "
            "their request, record confirmed reductions in `result.txt` in your workspace, "
            "and notify the CFO of every over-cap item."
        ),
    },
}


# Diversity variants — named slices of the registry.
# Counts: low=1, medium=3, high=8, extra_high=20
VARIANTS: dict[str, list[str]] = {
    "low": ["T01_dept_total_reply"],
    "medium": [
        "T01_dept_total_reply",
        "T02_personal_remaining_reply",
        "T03_dept_reduction",
    ],
    "high": [
        "T01_dept_total_reply",
        "T02_personal_remaining_reply",
        "T03_dept_reduction",
        "T04_dept_volume_discount_reply",
        "T05_dept_emergency_fund_reply",
        "T06_dept_restricted_categories_reply",
        "T08_dept_over_cap_escalation",
        "T10_dept_total_write_summary",
    ],
    "extra_high": list(TEMPLATES.keys()),
}
