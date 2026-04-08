"""Evaluation functions for oolong task.

Agents read a long context file and write their answer to answer.txt.
Evaluation uses normalized exact match (em_check) against ground-truth answers,
mirroring the original LOCA-bench QaEnv evaluation logic.
"""

import ast
import re
import string
from pathlib import Path
from typing import Any, Dict, Optional

import dspy


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = " ".join(s.split())
    return s


def _parse_reference_answers(raw: Any) -> list[str]:
    """Parse the reference answer field into a list of string answers.

    The dataset stores answers as string representations of Python lists,
    e.g. "['negative']" or "[5510]". Parse these to extract the actual values.
    """
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if isinstance(raw, (int, float)):
        return [str(raw)]
    s = str(raw).strip()
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [s]


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace_path = Path(workspace_dir)
    reference_answers = _parse_reference_answers(example["answer"])
    example_id = example.get("id", example.get("task_id"))

    answer_file = workspace_path / "answer.txt"
    if not answer_file.exists():
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "feedback": "answer.txt not found in workspace.",
            "agent_answer": "",
            "reference_answer": reference_answers[0] if reference_answers else "",
        }

    agent_answer = answer_file.read_text(encoding="utf-8").strip()
    normalized_agent = _normalize_answer(agent_answer)
    is_correct = any(
        _normalize_answer(ref) == normalized_agent or _normalize_answer(ref) in normalized_agent
        for ref in reference_answers
    )

    score = 1.0 if is_correct else 0.0
    if score == 1.0:
        feedback = "Correct."
    else:
        feedback = f"Incorrect. Expected {repr(reference_answers)}, got {repr(agent_answer)}."

    return {
        "workspace_dir": workspace_dir,
        "example_id": example_id,
        "score": score,
        "feedback": feedback,
        "agent_answer": agent_answer,
        "reference_answer": reference_answers[0] if reference_answers else "",
    }
