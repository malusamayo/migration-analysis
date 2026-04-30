import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import dspy


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def _money_matches(message: str, amount: float) -> bool:
    normalized = message.replace(",", "")
    candidates = {
        f"${amount:.2f}",
        f"${amount:.0f}",
        f"{amount:.2f}",
        f"{amount:.0f}",
    }
    return any(candidate.replace(",", "") in normalized for candidate in candidates)


def _thread_messages(state: dict, recipient: str, sender: str) -> list[str]:
    return [
        entry["message"]
        for entry in state["threads"].get(recipient, [])
        if entry["sender"] == sender
    ]


def _agent_contacted_all(state: dict, contacts: list[str]) -> bool:
    return all(_thread_messages(state, contact, "agent") for contact in contacts)


def _documents_accessed(state: dict, paths: list[str]) -> bool:
    accessed = {entry["path"] for entry in state["document_access_log"]}
    return all(path in accessed for path in paths)


def _decision_messages_correct(state: dict, expected: dict) -> bool:
    for recipient, decision in expected["decisions"].items():
        messages = _thread_messages(state, recipient, "agent")
        if not any(
            _money_matches(message, float(decision["total"]))
            and decision["status"] in _normalize_text(message)
            for message in messages
        ):
            return False
    return True


def _totals_only_messages_correct(state: dict, expected: dict) -> bool:
    for recipient, total in expected["manager_totals"].items():
        messages = _thread_messages(state, recipient, "agent")
        if not any(_money_matches(message, float(total)) for message in messages):
            return False
    return True


def _reduction_requests_made(state: dict, expected: dict) -> bool:
    for recipient in expected["reduction_contacts"]:
        messages = _thread_messages(state, recipient, "agent")
        if not any(
            keyword in _normalize_text(message)
            for message in messages
            for keyword in ("reduce", "remove", "cut")
        ):
            return False
    return True


def _result_file_correct(workspace: Path, expected: dict) -> bool:
    result_path = workspace / "result.txt"
    if not result_path.exists():
        return False
    lines = sorted(
        line.strip()
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return lines == expected["expected_result_lines"]


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    expected = _load_json(log_dir / "groundtruth" / "expected.json")
    state = _load_json(workspace / "local_db" / "agent_company" / "state.json")

    checkpoints: list[tuple[str, bool, int]] = []
    checkpoints.append(
        ("contacted required people", _agent_contacted_all(state, expected["required_contacts"]), 1)
    )
    checkpoints.append(
        ("read required documents", _documents_accessed(state, expected["required_documents"]), 1)
    )

    if expected["mode"] in {"department_budget_reply", "remaining_budget_reply"}:
        checkpoints.append(
            ("communicated correct totals and status", _decision_messages_correct(state, expected), 2)
        )
    elif expected["mode"] == "reduction_record":
        checkpoints.append(
            ("communicated initial totals", _totals_only_messages_correct(state, expected), 2)
        )
        checkpoints.append(
            ("requested reductions from over-budget department", _reduction_requests_made(state, expected), 1)
        )
        checkpoints.append(
            ("recorded exact equipment removals", _result_file_correct(workspace, expected), 2)
        )
    else:
        raise ValueError(f'Unknown mode: {expected["mode"]}')

    total = sum(weight for _, _, weight in checkpoints)
    score = sum(weight for _, passed, weight in checkpoints if passed)
    if checkpoints[-1][1]:
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
