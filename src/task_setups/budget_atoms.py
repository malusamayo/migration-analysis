"""Compositional atom library for budget-approval task variants.

Templates are declarative compositions over four atom categories:

- DOCUMENTS: which files appear in the mock filesystem
- CONTACT_QUERIES: which contact roles exist and what info they hold
- COMPUTATIONS: how to derive per-manager totals / per-department totals /
  remaining-budget / discount-adjusted totals / etc.
- ACTIONS: what the agent must produce (replies, reduction record file,
  approval file, summary file, CFO escalation, ...)

The MCP server's NPC reply logic routes on three fixed keyword classes
(`reduce/remove/cut`, `budget/spend/remaining`, `request/equipment/need/purchase`).
All non-manager roles therefore expose their info under the `budget_response`
slot; `list_contacts` carries the role hint so the agent can target the right
contact when its message contains a `budget`-class keyword.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import dspy


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

DOC_ROOT = "/Documents/Administrative Specialist"

DOC_PATHS = {
    "products": f"{DOC_ROOT}/products.csv",
    "dept_budgets": f"{DOC_ROOT}/department_budgets.csv",
    "reduction_policy": f"{DOC_ROOT}/reduction_policy.md",
    "headcount": f"{DOC_ROOT}/headcount.csv",
    "budget_pool": f"{DOC_ROOT}/budget_pool.md",
    "vendor_terms": f"{DOC_ROOT}/vendor_terms.md",
    "categories": f"{DOC_ROOT}/expense_categories.csv",
    "approval_policy": f"{DOC_ROOT}/approval_policy.md",
    "emergency_fund": f"{DOC_ROOT}/emergency_fund.md",
    "prior_quarter": f"{DOC_ROOT}/prior_quarter_spend.csv",
}


def _money(amount: float) -> str:
    return f"${amount:,.2f}"


def build_products_doc(example: dict) -> str:
    lines = ["product,unit_price"]
    for row in example["product_catalog"]:
        lines.append(f'{row["product"]},{row["unit_price"]:.2f}')
    return "\n".join(lines)


def build_dept_budgets_doc(example: dict) -> str:
    lines = ["department,budget"]
    for row in example["departments"]:
        lines.append(f'{row["name"]},{row["budget"]:.2f}')
    return "\n".join(lines)


def build_reduction_policy_doc(example: dict) -> str:
    return (
        "If a department exceeds budget, contact every manager in that department and ask "
        "them to reduce their expense. Record only confirmed removals in result.txt in the "
        "current workspace using the format `Name removed Equipment`."
    )


def build_headcount_doc(example: dict) -> str:
    lines = ["department,headcount"]
    for row in example["headcount"]:
        lines.append(f'{row["department"]},{row["headcount"]}')
    return "\n".join(lines)


def build_budget_pool_doc(example: dict) -> str:
    return f"# Total budget pool\n\nThe combined budget pool for this period is {_money(example['budget_pool'])}."


def build_vendor_terms_doc(example: dict) -> str:
    tiers = example["vendor_terms"]["tiers"]
    parts = ["# Vendor volume discount terms", ""]
    parts.append(
        "When a single department's pre-discount equipment order meets the threshold, "
        "the listed discount applies to that department's full order:"
    )
    parts.append("")
    for tier in tiers:
        parts.append(
            f'- Orders of {_money(tier["min_order"])} or more: '
            f'{int(round(tier["discount_pct"] * 100))}% off the whole order.'
        )
    return "\n".join(parts)


def build_categories_doc(example: dict) -> str:
    lines = ["product,category,restricted"]
    for row in example["categories"]:
        lines.append(
            f'{row["product"]},{row["category"]},{"yes" if row["restricted"] else "no"}'
        )
    return "\n".join(lines)


def build_approval_policy_doc(example: dict) -> str:
    pol = example["approval_policy"]
    return "\n".join(
        [
            "# Approval policy",
            "",
            f"Any single line item priced over {_money(pol['per_item_cap'])} must be "
            "escalated to the CFO before approval. Departments may not approve those "
            "items on their own.",
        ]
    )


def build_emergency_fund_doc(example: dict) -> str:
    lines = ["# Emergency fund", ""]
    lines.append(
        "Each department has access to an emergency reserve in addition to its standard "
        "budget. The reserve is added to the budget when comparing against the total cost."
    )
    lines.append("")
    lines.append("department,emergency_reserve")
    for row in example["emergency_fund"]:
        lines.append(f'{row["department"]},{row["reserve"]:.2f}')
    return "\n".join(lines)


def build_prior_quarter_doc(example: dict) -> str:
    lines = ["department,prior_quarter_spend"]
    for row in example["prior_quarter"]:
        lines.append(f'{row["department"]},{row["spend"]:.2f}')
    return "\n".join(lines)


DOC_BUILDERS: dict[str, Callable[[dict], str]] = {
    "products": build_products_doc,
    "dept_budgets": build_dept_budgets_doc,
    "reduction_policy": build_reduction_policy_doc,
    "headcount": build_headcount_doc,
    "budget_pool": build_budget_pool_doc,
    "vendor_terms": build_vendor_terms_doc,
    "categories": build_categories_doc,
    "approval_policy": build_approval_policy_doc,
    "emergency_fund": build_emergency_fund_doc,
    "prior_quarter": build_prior_quarter_doc,
}


# ---------------------------------------------------------------------------
# Contact queries — populate NPC profiles
# ---------------------------------------------------------------------------

# Each contact query atom describes one role that may appear in a template.
# `populate_state` writes the contact + profile into the MCP state. The MCP's
# build_reply only knows three keyword classes, so non-manager roles all use
# the `budget_response` slot with role-specific content. The agent can pick
# the right contact via `list_contacts` (each contact has a `role` field).


def _manager_request_string(manager: dict) -> str:
    pieces = [
        f'{request["quantity"]} {request["product"]}' for request in manager["requests"]
    ]
    return "My current equipment request is: " + ", ".join(pieces) + "."


def _set_manager_profile(state: dict, manager: dict) -> None:
    profile = state["npc_profiles"].setdefault(manager["name"], {})
    profile["request_response"] = _manager_request_string(manager)
    profile.setdefault(
        "fallback_response",
        "Please let me know what equipment-request information you need.",
    )


def populate_managers(state: dict, example: dict) -> None:
    for manager in example["managers"]:
        state["contacts"].append(
            {
                "name": manager["name"],
                "role": "manager",
                "department": manager["department"],
            }
        )
        _set_manager_profile(state, manager)
        state["threads"].setdefault(manager["name"], [])


def populate_finance(state: dict, example: dict) -> None:
    contact = example["finance_contact"]
    state["contacts"].append(
        {"name": contact["name"], "role": "finance", "department": "Finance"}
    )
    parts = []
    for person in example["people_budgets"]:
        parts.append(
            f'{person["name"]}: total budget {_money(person["total_budget"])}, '
            f'first-half spend {_money(person["spent"])}'
        )
    state["npc_profiles"][contact["name"]] = {
        "budget_response": "; ".join(parts) + ".",
        "fallback_response": (
            "Ask me about each employee's budget, spend, or remaining allowance."
        ),
    }
    state["threads"].setdefault(contact["name"], [])


def populate_hr(state: dict, example: dict) -> None:
    contact = example["hr_contact"]
    state["contacts"].append(
        {"name": contact["name"], "role": "hr", "department": "HR"}
    )
    parts = [f'{row["department"]}: {row["headcount"]}' for row in example["headcount"]]
    state["npc_profiles"][contact["name"]] = {
        "budget_response": (
            "Department headcount (use it to allocate the per-department budget by "
            "headcount share): " + "; ".join(parts) + "."
        ),
        "fallback_response": (
            "Ask me about department headcount or per-head budget allocation."
        ),
    }
    state["threads"].setdefault(contact["name"], [])


def populate_cfo(state: dict, example: dict) -> None:
    contact = example["cfo_contact"]
    state["contacts"].append(
        {"name": contact["name"], "role": "cfo", "department": "Executive"}
    )
    cap = example["approval_policy"]["per_item_cap"]
    state["npc_profiles"][contact["name"]] = {
        "budget_response": (
            f"Per-line-item approval cap is {_money(cap)}. Any single product priced "
            "above that must be escalated to me before approval."
        ),
        "fallback_response": (
            "Ask me about the per-item approval cap or escalate over-cap items."
        ),
    }
    state["threads"].setdefault(contact["name"], [])


def populate_vendor(state: dict, example: dict) -> None:
    contact = example["vendor_contact"]
    state["contacts"].append(
        {"name": contact["name"], "role": "vendor", "department": "External"}
    )
    tiers = example["vendor_terms"]["tiers"]
    parts = [
        f'orders over {_money(t["min_order"])} get '
        f'{int(round(t["discount_pct"] * 100))}% off'
        for t in tiers
    ]
    state["npc_profiles"][contact["name"]] = {
        "budget_response": "Current volume-discount tiers: " + "; ".join(parts) + ".",
        "fallback_response": (
            "Ask me about volume-discount tiers or remaining budget after a discount."
        ),
    }
    state["threads"].setdefault(contact["name"], [])


# Reductions are agent-initiated and reply via the `reduction_response` slot.
# This atom attaches that slot to the managers in over-budget departments.
def attach_reduction_responses(state: dict, expected: dict) -> None:
    plans: dict[str, list[str]] = expected.get("reduction_plan", {})
    for manager_name, removals in plans.items():
        profile = state["npc_profiles"].setdefault(manager_name, {})
        if removals:
            profile["reduction_response"] = "I can remove: " + ", ".join(removals) + "."
        else:
            profile["reduction_response"] = "I cannot reduce any items."


# ---------------------------------------------------------------------------
# Computations
# ---------------------------------------------------------------------------


def _price_map(example: dict) -> dict[str, float]:
    return {row["product"]: float(row["unit_price"]) for row in example["product_catalog"]}


def _per_manager_totals(example: dict) -> dict[str, float]:
    prices = _price_map(example)
    return {
        manager["name"]: round(
            sum(
                request["quantity"] * prices[request["product"]]
                for request in manager["requests"]
            ),
            2,
        )
        for manager in example["managers"]
    }


def _per_department_totals(example: dict, totals: dict[str, float]) -> dict[str, float]:
    dept_totals: defaultdict[str, float] = defaultdict(float)
    for manager in example["managers"]:
        dept_totals[manager["department"]] += totals[manager["name"]]
    return {k: round(v, 2) for k, v in dept_totals.items()}


def _dept_budget_map(example: dict) -> dict[str, float]:
    return {row["name"]: float(row["budget"]) for row in example["departments"]}


def compute_per_department(example: dict) -> dict:
    """Each manager labelled `under budget` / `exceed budget` based on dept total."""
    manager_totals = _per_manager_totals(example)
    dept_totals = _per_department_totals(example, manager_totals)
    dept_budgets = _dept_budget_map(example)
    decisions = {}
    for manager in example["managers"]:
        dept = manager["department"]
        status = "under budget" if dept_totals[dept] <= dept_budgets[dept] else "exceed budget"
        decisions[manager["name"]] = {"total": manager_totals[manager["name"]], "status": status}
    return {
        "manager_totals": manager_totals,
        "dept_totals": dept_totals,
        "dept_budgets": dept_budgets,
        "decisions": decisions,
    }


def compute_remaining_personal(example: dict) -> dict:
    """Each person labelled based on (total_budget - spent) vs request total."""
    manager_totals = _per_manager_totals(example)
    budgets = {row["name"]: row for row in example["people_budgets"]}
    decisions = {}
    for manager in example["managers"]:
        b = budgets[manager["name"]]
        remaining = round(float(b["total_budget"]) - float(b["spent"]), 2)
        status = "under budget" if manager_totals[manager["name"]] <= remaining else "exceed budget"
        decisions[manager["name"]] = {"total": manager_totals[manager["name"]], "status": status}
    return {"manager_totals": manager_totals, "decisions": decisions}


def compute_with_volume_discount(example: dict) -> dict:
    """Per-department total after a per-department volume discount tier is applied."""
    manager_totals = _per_manager_totals(example)
    dept_totals_raw = _per_department_totals(example, manager_totals)
    dept_budgets = _dept_budget_map(example)
    tiers = sorted(
        example["vendor_terms"]["tiers"], key=lambda t: t["min_order"], reverse=True
    )

    discounted = {}
    for dept, total in dept_totals_raw.items():
        applied = 0.0
        for tier in tiers:
            if total >= tier["min_order"]:
                applied = tier["discount_pct"]
                break
        discounted[dept] = round(total * (1 - applied), 2)

    decisions = {}
    for manager in example["managers"]:
        dept = manager["department"]
        status = (
            "under budget" if discounted[dept] <= dept_budgets[dept] else "exceed budget"
        )
        decisions[manager["name"]] = {
            "total": manager_totals[manager["name"]],
            "status": status,
        }
    return {
        "manager_totals": manager_totals,
        "dept_totals": dept_totals_raw,
        "dept_totals_discounted": discounted,
        "dept_budgets": dept_budgets,
        "decisions": decisions,
    }


def compute_headcount_allocated(example: dict) -> dict:
    """Each department's effective budget = global_pool * (headcount_share)."""
    manager_totals = _per_manager_totals(example)
    dept_totals = _per_department_totals(example, manager_totals)
    headcount = {row["department"]: int(row["headcount"]) for row in example["headcount"]}
    total_pool = float(example["budget_pool"])
    total_head = sum(headcount.values())
    dept_budgets = {
        dept: round(total_pool * headcount[dept] / total_head, 2) for dept in headcount
    }
    decisions = {}
    for manager in example["managers"]:
        dept = manager["department"]
        status = "under budget" if dept_totals[dept] <= dept_budgets[dept] else "exceed budget"
        decisions[manager["name"]] = {
            "total": manager_totals[manager["name"]],
            "status": status,
        }
    return {
        "manager_totals": manager_totals,
        "dept_totals": dept_totals,
        "dept_budgets": dept_budgets,
        "decisions": decisions,
    }


def compute_with_emergency_fund(example: dict) -> dict:
    """Effective budget = base + emergency reserve."""
    manager_totals = _per_manager_totals(example)
    dept_totals = _per_department_totals(example, manager_totals)
    base = _dept_budget_map(example)
    reserve = {row["department"]: float(row["reserve"]) for row in example["emergency_fund"]}
    dept_budgets = {dept: round(base[dept] + reserve.get(dept, 0.0), 2) for dept in base}
    decisions = {}
    for manager in example["managers"]:
        dept = manager["department"]
        status = "under budget" if dept_totals[dept] <= dept_budgets[dept] else "exceed budget"
        decisions[manager["name"]] = {
            "total": manager_totals[manager["name"]],
            "status": status,
        }
    return {
        "manager_totals": manager_totals,
        "dept_totals": dept_totals,
        "dept_budgets": dept_budgets,
        "decisions": decisions,
    }


def compute_restricted_filtered(example: dict) -> dict:
    """Restricted items are silently dropped before computing per-department totals."""
    prices = _price_map(example)
    restricted = {row["product"] for row in example["categories"] if row["restricted"]}
    manager_totals = {}
    for manager in example["managers"]:
        total = 0.0
        for request in manager["requests"]:
            if request["product"] in restricted:
                continue
            total += request["quantity"] * prices[request["product"]]
        manager_totals[manager["name"]] = round(total, 2)
    dept_totals = _per_department_totals(example, manager_totals)
    dept_budgets = _dept_budget_map(example)
    decisions = {}
    for manager in example["managers"]:
        dept = manager["department"]
        status = "under budget" if dept_totals[dept] <= dept_budgets[dept] else "exceed budget"
        decisions[manager["name"]] = {
            "total": manager_totals[manager["name"]],
            "status": status,
        }
    return {
        "manager_totals": manager_totals,
        "dept_totals": dept_totals,
        "dept_budgets": dept_budgets,
        "restricted_products": sorted(restricted),
        "decisions": decisions,
    }


def compute_over_cap_items(example: dict) -> dict:
    """Department totals as usual + a list of `(manager, product)` items over the cap."""
    base = compute_per_department(example)
    prices = _price_map(example)
    cap = float(example["approval_policy"]["per_item_cap"])
    over_cap: list[tuple[str, str]] = []
    for manager in example["managers"]:
        for request in manager["requests"]:
            if prices[request["product"]] > cap:
                over_cap.append((manager["name"], request["product"]))
    base["over_cap_items"] = over_cap
    base["per_item_cap"] = cap
    return base


COMPUTATIONS: dict[str, Callable[[dict], dict]] = {
    "per_department": compute_per_department,
    "remaining_personal": compute_remaining_personal,
    "with_volume_discount": compute_with_volume_discount,
    "headcount_allocated": compute_headcount_allocated,
    "with_emergency_fund": compute_with_emergency_fund,
    "restricted_filtered": compute_restricted_filtered,
    "over_cap_items": compute_over_cap_items,
}


# ---------------------------------------------------------------------------
# Reduction plan (deterministic, shared across reduction-bearing templates)
# ---------------------------------------------------------------------------


def build_adjustment_plan(example: dict, dept_totals: dict[str, float], dept_budgets: dict[str, float]) -> dict[str, list[str]]:
    price_by_product = _price_map(example)
    managers_by_department: defaultdict[str, list[dict]] = defaultdict(list)
    for manager in example["managers"]:
        managers_by_department[manager["department"]].append(manager)

    plans: dict[str, list[str]] = {m["name"]: [] for m in example["managers"]}
    for dept, total in dept_totals.items():
        budget = dept_budgets[dept]
        overflow = round(total - budget, 2)
        if overflow <= 0:
            continue
        remaining = overflow
        for manager in sorted(managers_by_department[dept], key=lambda m: m["name"], reverse=True):
            for request in sorted(
                manager["requests"],
                key=lambda r: price_by_product[r["product"]],
                reverse=True,
            ):
                if remaining <= 0:
                    break
                removable = max(1, request["quantity"] // 2)
                unit_price = price_by_product[request["product"]]
                needed = min(
                    request["quantity"],
                    max(1, int((remaining + unit_price - 1) // unit_price)),
                )
                remove = min(removable, needed)
                if remove <= 0:
                    continue
                plans[manager["name"]].extend([request["product"]] * remove)
                remaining = round(remaining - remove * unit_price, 2)
            if remaining <= 0:
                break
    return plans


# ---------------------------------------------------------------------------
# Action eval checks (the heart of the eval; each action contributes one
# checkpoint with a name and weight)
# ---------------------------------------------------------------------------


_REDUCE_VERBS = ("reduce", "remove", "cut")


def _thread_msgs(state: dict, recipient: str, sender: str) -> list[str]:
    return [
        e["message"]
        for e in state["threads"].get(recipient, [])
        if e["sender"] == sender
    ]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _money_in(message: str, amount: float) -> bool:
    normalized = message.replace(",", "")
    candidates = {
        f"${amount:.2f}",
        f"${amount:.0f}",
        f"{amount:.2f}",
        f"{amount:.0f}",
    }
    return any(c.replace(",", "") in normalized for c in candidates)


def _llm_status_match(lm: dspy.LM, message: str, expected_status: str) -> bool:
    prompt = (
        "Determine whether the message communicates the same budget status as the expected "
        "status.\n"
        f"Expected status: {expected_status}\n"
        f"Message: {message}\n\n"
        "Treat close paraphrases as equivalent. For example, 'exceeding budget' and "
        "'over budget' are equivalent to 'exceed budget', while 'within budget' is "
        "equivalent to 'under budget'. Respond with 'yes' or 'no' only."
    )
    response = lm(messages=[{"role": "user", "content": prompt}])
    return response[0].strip().lower().startswith("yes")


def _status_in(message: str, expected_status: str, lm: Optional[dspy.LM]) -> bool:
    normalized = _normalize(message)
    if expected_status in normalized:
        return True
    equivalents = {
        "exceed budget": ("exceeding budget", "exceeds budget", "over budget", "above budget"),
        "under budget": ("within budget", "below budget"),
    }
    if any(p in normalized for p in equivalents.get(expected_status, ())):
        return True
    if lm is None:
        return False
    return _llm_status_match(lm, message, expected_status)


def check_reply_total_status(
    state: dict, expected: dict, workspace: Path, lm: Optional[dspy.LM]
) -> bool:
    for recipient, decision in expected["decisions"].items():
        messages = _thread_msgs(state, recipient, "agent")
        ok = any(
            _money_in(msg, float(decision["total"]))
            and _status_in(msg, decision["status"], lm)
            for msg in messages
        )
        if not ok:
            return False
    return True


def check_reply_total_only(
    state: dict, expected: dict, workspace: Path, lm: Optional[dspy.LM]
) -> bool:
    for recipient, total in expected["manager_totals"].items():
        messages = _thread_msgs(state, recipient, "agent")
        if not any(_money_in(msg, float(total)) for msg in messages):
            return False
    return True


def check_reduction_requests(
    state: dict, expected: dict, workspace: Path, lm: Optional[dspy.LM]
) -> bool:
    for recipient in expected.get("reduction_contacts", []):
        messages = _thread_msgs(state, recipient, "agent")
        if not any(
            verb in _normalize(msg) for msg in messages for verb in _REDUCE_VERBS
        ):
            return False
    return True


def check_reduction_record(
    state: dict, expected: dict, workspace: Path, lm: Optional[dspy.LM]
) -> bool:
    path = workspace / "result.txt"
    if not path.exists():
        return False
    lines = sorted(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return lines == expected["expected_result_lines"]


def check_escalate_to_cfo(
    state: dict, expected: dict, workspace: Path, lm: Optional[dspy.LM]
) -> bool:
    cfo = expected.get("cfo_contact_name")
    if not cfo:
        return False
    messages = _thread_msgs(state, cfo, "agent")
    if not messages:
        return False
    # Every product that is over the cap must be named in some message to the CFO.
    over_cap_products = {p for _, p in expected.get("over_cap_items", [])}
    joined = " | ".join(_normalize(m) for m in messages)
    return all(p.lower() in joined for p in over_cap_products)


def check_write_approval_txt(
    state: dict, expected: dict, workspace: Path, lm: Optional[dspy.LM]
) -> bool:
    path = workspace / "approval.txt"
    if not path.exists():
        return False
    actual = sorted(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return actual == expected["expected_approval_lines"]


def check_write_summary_txt(
    state: dict, expected: dict, workspace: Path, lm: Optional[dspy.LM]
) -> bool:
    path = workspace / "summary.txt"
    if not path.exists():
        return False
    content = _normalize(path.read_text(encoding="utf-8"))
    for dept, total in expected["dept_totals"].items():
        if not _money_in(content, float(total)):
            return False
        if dept.lower() not in content:
            return False
    return True


@dataclass
class ActionSpec:
    name: str
    weight: int
    check: Callable[[dict, dict, Path, Optional[dspy.LM]], bool]
    terminal: bool = False  # if true, score is 100% when this passes


ACTIONS: dict[str, ActionSpec] = {
    "reply_total_status": ActionSpec(
        name="communicated correct totals and status",
        weight=2,
        check=check_reply_total_status,
    ),
    "reply_total_only": ActionSpec(
        name="communicated initial totals",
        weight=2,
        check=check_reply_total_only,
    ),
    "reduction_requests": ActionSpec(
        name="requested reductions from over-budget departments",
        weight=1,
        check=check_reduction_requests,
    ),
    "reduction_record": ActionSpec(
        name="recorded exact equipment removals",
        weight=2,
        check=check_reduction_record,
        terminal=True,
    ),
    "escalate_to_cfo": ActionSpec(
        name="escalated over-cap items to the CFO",
        weight=2,
        check=check_escalate_to_cfo,
    ),
    "write_approval_txt": ActionSpec(
        name="wrote approval.txt with correct decisions",
        weight=2,
        check=check_write_approval_txt,
    ),
    "write_summary_txt": ActionSpec(
        name="wrote summary.txt with correct department totals",
        weight=2,
        check=check_write_summary_txt,
    ),
}


# ---------------------------------------------------------------------------
# Build expected outcomes for an example given a template spec
# ---------------------------------------------------------------------------


def required_documents(template_docs: list[str]) -> list[str]:
    return [DOC_PATHS[d] for d in template_docs]


def required_contacts(example: dict, template_queries: list[str]) -> list[str]:
    contacts = [manager["name"] for manager in example["managers"]]
    if "finance" in template_queries:
        contacts.append(example["finance_contact"]["name"])
    if "hr" in template_queries:
        contacts.append(example["hr_contact"]["name"])
    if "cfo" in template_queries:
        contacts.append(example["cfo_contact"]["name"])
    if "vendor" in template_queries:
        contacts.append(example["vendor_contact"]["name"])
    return contacts


def build_expected(example: dict, template: dict) -> dict:
    compute = COMPUTATIONS[template["compute"]]
    result = compute(example)
    expected = {
        "template_id": template["id"],
        "required_contacts": required_contacts(example, template["queries"]),
        "required_documents": required_documents(template["docs"]),
        "manager_totals": result["manager_totals"],
        "decisions": result.get("decisions", {}),
    }
    if "dept_totals" in result:
        expected["dept_totals"] = result["dept_totals"]
    if "dept_budgets" in result:
        expected["dept_budgets"] = result["dept_budgets"]
    if "dept_totals_discounted" in result:
        expected["dept_totals_discounted"] = result["dept_totals_discounted"]
    if "restricted_products" in result:
        expected["restricted_products"] = result["restricted_products"]
    if "over_cap_items" in result:
        expected["over_cap_items"] = result["over_cap_items"]
        expected["per_item_cap"] = result["per_item_cap"]

    # Reduction plan + record lines (only when a reduction action is requested)
    if "reduction_record" in template["actions"] or "reduction_requests" in template["actions"]:
        plans = build_adjustment_plan(
            example,
            expected.get("dept_totals", {}),
            expected.get("dept_budgets", {}),
        )
        expected["reduction_plan"] = plans
        dept_totals = expected["dept_totals"]
        dept_budgets = expected["dept_budgets"]
        expected["reduction_contacts"] = [
            m["name"]
            for m in example["managers"]
            if dept_totals[m["department"]] > dept_budgets[m["department"]]
        ]
        expected["expected_result_lines"] = sorted(
            f"{name} removed {product}"
            for name, products in plans.items()
            for product in products
        )

    # Approval file lines: "ManagerName: under budget / exceed budget @ $total"
    if "write_approval_txt" in template["actions"]:
        expected["expected_approval_lines"] = sorted(
            f'{name}: {d["status"]} @ {_money(d["total"])}'
            for name, d in expected["decisions"].items()
        )

    if "escalate_to_cfo" in template["actions"]:
        expected["cfo_contact_name"] = example["cfo_contact"]["name"]

    return expected
