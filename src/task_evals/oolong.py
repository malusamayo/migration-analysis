"""Evaluation functions for oolong task.

Agents read a long D&D transcript from context.txt and write their answer to
answer.txt. Evaluation is exact string match against the ground-truth answer.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import dspy


def _extract_boxed(text: str) -> str:
    """Extract value from \\boxed{...} if present, otherwise return stripped text."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    """Evaluate a single oolong instance.

    Reads answer.txt from the workspace and compares it (exact match) to the
    ground-truth answer stored in the example dict.

    Args:
        workspace_dir: Path to the workspace directory.
        example: The original example dict, must contain an "answer" key.
        lm: Unused; kept for interface compatibility.

    Returns:
        Dict with keys: workspace_dir, example_id, score, feedback, agent_answer,
        reference_answer.
    """
    workspace_path = Path(workspace_dir)
    reference_answer = _extract_boxed(str(example["answer"]))
    example_id = example.get("id", example.get("task_id"))

    answer_file = workspace_path / "answer.txt"
    if not answer_file.exists():
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "feedback": "answer.txt not found in workspace.",
            "agent_answer": "",
            "reference_answer": reference_answer,
        }

    agent_answer = _extract_boxed(answer_file.read_text(encoding="utf-8"))

    score = 1.0 if agent_answer == reference_answer else 0.0
    if score == 1.0:
        feedback = "Correct."
    else:
        feedback = f"Incorrect. Expected {repr(reference_answer)}, got {repr(agent_answer)}."

    return {
        "workspace_dir": workspace_dir,
        "example_id": example_id,
        "score": score,
        "feedback": feedback,
        "agent_answer": agent_answer,
        "reference_answer": reference_answer,
    }
