"""Template-driven eval for the budget-approval task family.

Two universal checkpoints (`contacted required people`, `read required
documents`) plus one checkpoint per action atom referenced by the example's
template. Scoring rule preserved from the original eval: weighted average,
with the *last* checkpoint acting as a terminal gate — if it passes, the
example scores 100% regardless of partial checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import dspy

from ..task_setups.budget_atoms import ACTIONS
from ..task_setups.budget_templates import TEMPLATES


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _agent_contacted_all(state: dict, contacts: list[str]) -> bool:
    threads = state["threads"]
    return all(
        any(entry["sender"] == "agent" for entry in threads.get(name, []))
        for name in contacts
    )


def _documents_accessed(state: dict, paths: list[str]) -> bool:
    accessed = {entry["path"] for entry in state["document_access_log"]}
    return all(path in accessed for path in paths)


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    expected = _load_json(log_dir / "groundtruth" / "expected.json")
    try:
        state = _load_json(workspace / "local_db" / "agent_company" / "state.json")
    except json.JSONDecodeError as e:
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": f"Invalid state.json: {e}",
        }

    template = TEMPLATES[expected["template_id"]]

    checkpoints: list[tuple[str, bool, int]] = []
    checkpoints.append(
        (
            "contacted required people",
            _agent_contacted_all(state, expected["required_contacts"]),
            1,
        )
    )
    checkpoints.append(
        (
            "read required documents",
            _documents_accessed(state, expected["required_documents"]),
            1,
        )
    )

    for action_id in template["actions"]:
        spec = ACTIONS[action_id]
        passed = spec.check(state, expected, workspace, lm)
        checkpoints.append((spec.name, passed, spec.weight))

    total = sum(weight for _, _, weight in checkpoints)
    score = sum(weight for _, passed, weight in checkpoints if passed)

    # Terminal gate: if the last action's check is a terminal one and passes,
    # award full credit (matches original behavior for reduction tasks).
    last_action_id = template["actions"][-1] if template["actions"] else None
    if last_action_id and ACTIONS[last_action_id].terminal and checkpoints[-1][1]:
        score = total

    if score == total:
        feedback = "All checkpoints passed."
    else:
        failed = [name for name, passed, _ in checkpoints if not passed]
        feedback = "Failed checkpoints: " + ", ".join(failed)

    return {
        "workspace_dir": workspace_dir,
        "score": score / total,
        "feedback": feedback,
    }
