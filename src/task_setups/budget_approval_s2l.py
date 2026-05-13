"""Template-driven setup for the budget-approval task family.

The example dict carries a `template_id` field selecting one of the templates
in `budget_templates.TEMPLATES`. The template spec drives:

- which documents are materialized into the mock filesystem
- which contacts (managers + finance/HR/CFO/vendor) are listed in the MCP and
  what their NPC profile responses look like
- which computation produces the `expected.json` ground truth
- which action checkpoints the eval will run

All variants (`low` / `medium` / `high` / `extra_high`) share this file; the
only thing that changes between variants is which subset of templates appears
in the generated data file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .budget_atoms import (
    DOC_BUILDERS,
    DOC_PATHS,
    attach_reduction_responses,
    build_expected,
    populate_cfo,
    populate_finance,
    populate_hr,
    populate_managers,
    populate_vendor,
)
from .budget_templates import TEMPLATES


# Legacy bridge: pre-template-refactor example files used a `mode` field with
# one of three string values. Each maps cleanly onto a current template — the
# atoms, computations, docs, contacts, and eval weights are identical to the
# original pipeline, so old data files run as-is without regeneration. New
# examples carry `template_id` directly and skip this mapping.
_LEGACY_MODE_MAP = {
    "department_budget_reply": "T01_dept_total_reply",
    "remaining_budget_reply": "T02_personal_remaining_reply",
    "reduction_record": "T03_dept_reduction",
}


def _template_for(example: dict) -> dict:
    template_id = example.get("template_id")
    if template_id is None:
        template_id = _LEGACY_MODE_MAP[example["mode"]]
    return TEMPLATES[template_id]


def build_prompt(example: dict) -> str:
    """Goal-level prompt taken from the template spec."""
    return _template_for(example)["goal_prompt"]


def _build_documents(example: dict, template: dict) -> dict[str, str]:
    return {DOC_PATHS[d]: DOC_BUILDERS[d](example) for d in template["docs"]}


def _populate_contacts(state: dict, example: dict, template: dict) -> None:
    populate_managers(state, example)
    for query in template["queries"]:
        if query == "finance":
            populate_finance(state, example)
        elif query == "hr":
            populate_hr(state, example)
        elif query == "cfo":
            populate_cfo(state, example)
        elif query == "vendor":
            populate_vendor(state, example)
        else:
            raise ValueError(f"Unknown contact query atom: {query!r}")


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    template = _template_for(example)
    workspace = Path(workspace_dir)
    data_dir = workspace / "local_db" / "agent_company"
    data_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "contacts": [],
        "documents": _build_documents(example, template),
        "document_access_log": [],
        "threads": {},
        "message_log": [],
        "npc_profiles": {},
        "tick": 0,
    }
    _populate_contacts(state, example, template)

    expected = build_expected(example, template)
    if "reduction_record" in template["actions"] or "reduction_requests" in template["actions"]:
        attach_reduction_responses(state, expected)

    state_path = data_dir / "state.json"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # Pre-create artifact files the agent is expected to write so it's clear
    # which workspace files belong to the task.
    if "reduction_record" in template["actions"]:
        (workspace / "result.txt").write_text("", encoding="utf-8")
    if "write_approval_txt" in template["actions"]:
        (workspace / "approval.txt").write_text("", encoding="utf-8")
    if "write_summary_txt" in template["actions"]:
        (workspace / "summary.txt").write_text("", encoding="utf-8")

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
    loca_root = (
        Path("/loca-bench")
        if docker_workspace
        else Path(__file__).parent.parent.parent / "LOCA-bench"
    )
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
