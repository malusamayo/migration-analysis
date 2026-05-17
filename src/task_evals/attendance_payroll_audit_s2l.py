"""Template-driven eval for the attendance-payroll-audit task family.

Each template's `actions` list determines which deliverable artifacts get
checked. Scoring rule: weighted average across checkpoints; if the final
action is `terminal` and passes, the example scores 100% (matches the
budget-approval terminal-gate convention).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import dspy

from src.task_setups.attendance_atoms import ACTIONS
from src.task_setups.attendance_templates import TEMPLATES


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> dict[str, Any]:
    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    expected = _load_json(log_dir / "groundtruth" / "expected.json")
    template = TEMPLATES[expected["template_id"]]

    state: dict = {}
    checkpoints: list[tuple[str, bool, int]] = []
    for action_id in template["actions"]:
        spec = ACTIONS[action_id]
        passed = bool(spec.check(state, expected, workspace, lm))
        checkpoints.append((spec.name, passed, spec.weight))

    total = sum(weight for _, _, weight in checkpoints)
    score = sum(weight for _, passed, weight in checkpoints if passed)

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
        "score": score / total if total else 0.0,
        "feedback": feedback,
        "example_id": example.get("id"),
    }
