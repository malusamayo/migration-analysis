"""Evaluation for browsecompplus task.

Agents write their final answer to answer.txt.
Evaluation uses an LLM judge following the BrowseComp grading methodology.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import dspy


EVALUATION_PROMPT = """\
Judge whether the following [response] to [question] is correct or not \
based on the precise and unambiguous [correct_answer].

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

reasoning: Explain briefly whether the response matches the correct answer.

correct: Answer 'yes' if the response matches the correct answer \
(minor spelling variants are fine for names), 'no' otherwise.

Score (1=correct, 0=wrong):"""


def _parse_score(response: str) -> int:
    if re.search(r"\byes\b", response, re.IGNORECASE):
        return 1
    if re.search(r"\bno\b", response, re.IGNORECASE):
        return 0
    match = re.search(r"\b([01])\b", response)
    if match:
        return int(match.group(1))
    return 0


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace_path = Path(workspace_dir)
    example_id = example.get("query_id", example.get("id"))
    reference_answer = example["answer"]
    question = example["query"]

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

    agent_answer = answer_file.read_text(encoding="utf-8").strip()

    if not agent_answer:
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "feedback": "Answer is empty.",
            "agent_answer": agent_answer,
            "reference_answer": reference_answer,
        }

    if lm is None:
        raise ValueError("browsecompplus evaluation requires an LM judge (eval_lm must be set)")

    prompt = EVALUATION_PROMPT.format(
        question=question,
        response=agent_answer,
        correct_answer=reference_answer,
    )

    response = lm(messages=[{"role": "user", "content": prompt}])
    response_text = response[0] if isinstance(response, list) else str(response)

    score = float(_parse_score(response_text))

    return {
        "workspace_dir": workspace_dir,
        "example_id": example_id,
        "score": score,
        "feedback": f"Correctness: {int(score)} (raw: {response_text.strip()!r})",
        "agent_answer": agent_answer,
        "reference_answer": reference_answer,
    }
