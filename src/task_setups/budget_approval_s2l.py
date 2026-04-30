import json
import os
from collections import defaultdict
from pathlib import Path


PRODUCT_DOC_PATH = "/Documents/Administrative Specialist/products.csv"
DEPARTMENT_BUDGET_DOC_PATH = "/Documents/Administrative Specialist/department_budgets.csv"
REDUCTION_POLICY_DOC_PATH = "/Documents/Administrative Specialist/reduction_policy.md"

MODE_DEPARTMENT = "department_budget_reply"
MODE_REMAINING = "remaining_budget_reply"
MODE_REDUCTION = "reduction_record"


def format_money(amount: float) -> str:
    return f"${amount:,.2f}"


def build_prompt(example: dict) -> str:
    mode = example["mode"]
    if mode == MODE_DEPARTMENT:
        lines = [
            "1. Collect equipment requests (name and quantity) from these department managers by messaging them with the Agent Company communication tools:",
        ]
        for manager in example["managers"]:
            lines.append(f'   - {manager["name"]} ({manager["department"]})')
        lines.extend(
            [
                "",
                f"2. Retrieve equipment prices from `{PRODUCT_DOC_PATH}` using the Agent Company document tools.",
                f"3. Retrieve department budgets from `{DEPARTMENT_BUDGET_DOC_PATH}`.",
                "4. Send a reply to every department manager that includes:",
                "   - the exact total cost of that manager's request",
                "   - whether the department is `under budget` or `exceed budget` after aggregating all requests in that department",
            ]
        )
        return "\n".join(lines)

    if mode == MODE_REMAINING:
        lines = [
            "1. Collect equipment requests (name and quantity) from these employees by messaging them with the Agent Company communication tools:",
        ]
        for manager in example["managers"]:
            lines.append(f'   - {manager["name"]}')
        lines.extend(
            [
                "",
                f"2. Retrieve equipment prices from `{PRODUCT_DOC_PATH}` using the Agent Company document tools.",
                f'3. Ask {example["finance_contact"]["name"]} for each employee\'s total budget and first-half spend.',
                "4. Reply to every employee with:",
                "   - the exact total cost of that employee's request",
                "   - whether the employee is `under budget` or `exceed budget` based on the remaining budget",
            ]
        )
        return "\n".join(lines)

    if mode == MODE_REDUCTION:
        lines = [
            "1. Collect equipment requests (name and quantity) from these department managers by messaging them with the Agent Company communication tools:",
        ]
        for manager in example["managers"]:
            lines.append(f'   - {manager["name"]} ({manager["department"]})')
        lines.extend(
            [
                "",
                f"2. Retrieve equipment prices from `{PRODUCT_DOC_PATH}` using the Agent Company document tools.",
                f"3. Retrieve department budgets from `{DEPARTMENT_BUDGET_DOC_PATH}` and the policy note from `{REDUCTION_POLICY_DOC_PATH}`.",
                "4. Tell each manager the exact total cost of that manager's current request.",
                "5. If a department exceeds budget after aggregating all requests in that department, contact every manager in that department and ask them to reduce their expense.",
                "6. Record every confirmed equipment change in `result.txt` in the current workspace using exactly one line per change in the format `Name removed Equipment`.",
            ]
        )
        return "\n".join(lines)

    raise ValueError(f"Unknown mode: {mode}")


def build_product_document(example: dict) -> str:
    lines = ["product,unit_price"]
    for row in example["product_catalog"]:
        lines.append(f'{row["product"]},{row["unit_price"]:.2f}')
    return "\n".join(lines)


def build_department_budget_document(example: dict) -> str:
    lines = ["department,budget"]
    for row in example["departments"]:
        lines.append(f'{row["name"]},{row["budget"]:.2f}')
    return "\n".join(lines)


def build_reduction_policy_document() -> str:
    return (
        "If a department exceeds budget, contact every manager in that department and ask "
        "them to reduce their expense. Record only confirmed removals in result.txt in the "
        "current workspace using the format `Name removed Equipment`."
    )


def build_request_response(manager: dict) -> str:
    pieces = [
        f'{request["quantity"]} {request["product"]}'
        for request in manager["requests"]
    ]
    return "My current equipment request is: " + ", ".join(pieces) + "."


def build_budget_response(example: dict) -> str:
    parts = []
    for person in example["people_budgets"]:
        parts.append(
            f'{person["name"]}: total budget {format_money(person["total_budget"])}, '
            f'first-half spend {format_money(person["spent"])}'
        )
    return "; ".join(parts) + "."


def build_adjustment_plan(example: dict) -> dict[str, list[str]]:
    price_by_product = {
        row["product"]: float(row["unit_price"])
        for row in example["product_catalog"]
    }
    managers_by_department: defaultdict[str, list[dict]] = defaultdict(list)
    for manager in example["managers"]:
        managers_by_department[manager["department"]].append(manager)

    plans: dict[str, list[str]] = {manager["name"]: [] for manager in example["managers"]}
    for department in example["departments"]:
        name = department["name"]
        budget = float(department["budget"])
        managers = managers_by_department[name]
        total = sum(
            request["quantity"] * price_by_product[request["product"]]
            for manager in managers
            for request in manager["requests"]
        )
        overflow = round(total - budget, 2)
        if overflow <= 0:
            continue
        remaining_overflow = overflow
        for manager in sorted(managers, key=lambda item: item["name"], reverse=True):
            for request in sorted(
                manager["requests"],
                key=lambda item: price_by_product[item["product"]],
                reverse=True,
            ):
                if remaining_overflow <= 0:
                    break
                removable_units = max(1, request["quantity"] // 2)
                unit_price = price_by_product[request["product"]]
                needed_units = min(
                    request["quantity"],
                    max(1, int((remaining_overflow + unit_price - 1) // unit_price)),
                )
                remove_units = min(removable_units, needed_units)
                if remove_units <= 0:
                    continue
                plans[manager["name"]].extend([request["product"]] * remove_units)
                remaining_overflow = round(remaining_overflow - remove_units * unit_price, 2)
            if remaining_overflow <= 0:
                break
    return plans


def build_expected_outcomes(example: dict) -> dict:
    price_by_product = {
        row["product"]: float(row["unit_price"])
        for row in example["product_catalog"]
    }
    manager_totals = {
        manager["name"]: round(
            sum(
                request["quantity"] * price_by_product[request["product"]]
                for request in manager["requests"]
            ),
            2,
        )
        for manager in example["managers"]
    }

    expected = {
        "mode": example["mode"],
        "required_contacts": [manager["name"] for manager in example["managers"]],
        "required_documents": [PRODUCT_DOC_PATH],
        "manager_totals": manager_totals,
        "decisions": {},
        "reduction_contacts": [],
        "expected_result_lines": [],
    }

    if example["mode"] == MODE_DEPARTMENT:
        department_totals: defaultdict[str, float] = defaultdict(float)
        department_budgets = {
            department["name"]: float(department["budget"])
            for department in example["departments"]
        }
        for manager in example["managers"]:
            department_totals[manager["department"]] += manager_totals[manager["name"]]
        expected["required_documents"].append(DEPARTMENT_BUDGET_DOC_PATH)
        for manager in example["managers"]:
            status = (
                "under budget"
                if department_totals[manager["department"]] <= department_budgets[manager["department"]]
                else "exceed budget"
            )
            expected["decisions"][manager["name"]] = {
                "total": manager_totals[manager["name"]],
                "status": status,
            }
        return expected

    if example["mode"] == MODE_REMAINING:
        expected["required_contacts"].append(example["finance_contact"]["name"])
        budget_rows = {
            person["name"]: person
            for person in example["people_budgets"]
        }
        for manager in example["managers"]:
            budget_row = budget_rows[manager["name"]]
            remaining = round(
                float(budget_row["total_budget"]) - float(budget_row["spent"]),
                2,
            )
            status = (
                "under budget"
                if manager_totals[manager["name"]] <= remaining
                else "exceed budget"
            )
            expected["decisions"][manager["name"]] = {
                "total": manager_totals[manager["name"]],
                "status": status,
            }
        return expected

    if example["mode"] == MODE_REDUCTION:
        department_totals: defaultdict[str, float] = defaultdict(float)
        department_budgets = {
            department["name"]: float(department["budget"])
            for department in example["departments"]
        }
        for manager in example["managers"]:
            department_totals[manager["department"]] += manager_totals[manager["name"]]
        expected["required_documents"].extend(
            [DEPARTMENT_BUDGET_DOC_PATH, REDUCTION_POLICY_DOC_PATH]
        )
        plans = build_adjustment_plan(example)
        expected["reduction_contacts"] = [
            manager["name"]
            for manager in example["managers"]
            if department_totals[manager["department"]] > department_budgets[manager["department"]]
        ]
        expected["reduction_plan"] = plans
        expected["expected_result_lines"] = sorted(
            f"{manager_name} removed {product}"
            for manager_name, products in plans.items()
            for product in products
        )
        return expected

    raise ValueError(f'Unknown mode: {example["mode"]}')


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    workspace = Path(workspace_dir)
    data_dir = workspace / "local_db" / "agent_company"
    data_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "contacts": [],
        "documents": {
            PRODUCT_DOC_PATH: build_product_document(example),
        },
        "document_access_log": [],
        "threads": {},
        "message_log": [],
        "npc_profiles": {},
        "tick": 0,
    }

    if example["mode"] in {MODE_DEPARTMENT, MODE_REDUCTION}:
        state["documents"][DEPARTMENT_BUDGET_DOC_PATH] = build_department_budget_document(example)
    if example["mode"] == MODE_REDUCTION:
        state["documents"][REDUCTION_POLICY_DOC_PATH] = build_reduction_policy_document()

    expected = build_expected_outcomes(example)

    for manager in example["managers"]:
        state["contacts"].append(
            {
                "name": manager["name"],
                "role": "manager",
                "department": manager["department"],
            }
        )
        profile = {
            "request_response": build_request_response(manager),
            "fallback_response": "Please let me know what information you need.",
        }
        if example["mode"] == MODE_REDUCTION:
            removed_products = expected["reduction_plan"][manager["name"]]
            if removed_products:
                profile["reduction_response"] = (
                    "I can remove: " + ", ".join(removed_products) + "."
                )
            else:
                profile["reduction_response"] = "I cannot reduce any items."
        state["npc_profiles"][manager["name"]] = profile
        state["threads"][manager["name"]] = []

    if example["mode"] == MODE_REMAINING:
        finance_name = example["finance_contact"]["name"]
        state["contacts"].append(
            {
                "name": finance_name,
                "role": "finance",
                "department": example["finance_contact"]["department"],
            }
        )
        state["npc_profiles"][finance_name] = {
            "budget_response": build_budget_response(example),
            "fallback_response": "Ask me about employee budgets and spend when you are ready.",
        }
        state["threads"][finance_name] = []

    state_path = data_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    if example["mode"] == MODE_REDUCTION:
        (workspace / "result.txt").write_text("", encoding="utf-8")

    groundtruth_dir = Path(log_dir) / "groundtruth"
    groundtruth_dir.mkdir(parents=True, exist_ok=True)
    (groundtruth_dir / "expected.json").write_text(
        json.dumps(expected, indent=2),
        encoding="utf-8",
    )


def setup_proposer_workspace(workspace_dir: str) -> None:
    Path(workspace_dir, "local_db", "agent_company").mkdir(parents=True, exist_ok=True)


def get_mcp_config(workspace_dir: str) -> dict:
    workspace = Path(workspace_dir).resolve()
    data_dir = workspace / "local_db" / "agent_company"
    docker_workspace = str(workspace).startswith("/workspace/")
    loca_root = Path("/loca-bench") if docker_workspace else Path(__file__).parent.parent.parent / "LOCA-bench"
    server_script = loca_root / "mcp_convert" / "mcps" / "agent_company" / "server.py"
    project_root = loca_root / "mcp_convert"

    if docker_workspace or os.environ.get("LOCA_BENCH_PATH"):
        command = "/workspace/.venv/bin/python"
        args = [str(server_script)]
    else:
        args = [
            "--directory",
            str(project_root),
            "run",
            "python",
            str(server_script),
        ]
        command = "uv"

    return {
        "mcpServers": {
            "agent_company": {
                "command": command,
                "args": args,
                "env": {
                    "AGENT_COMPANY_DATA_DIR": str(data_dir),
                    "LOCA_QUIET": "1",
                },
            }
        }
    }
