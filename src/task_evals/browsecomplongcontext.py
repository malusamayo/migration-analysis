"""Evaluation functions for browsecomplongcontext task.

Agents read pre-fetched web pages from context.txt and write their answer to answer.txt.
Evaluation uses an LLM judge following the BrowseComp grading methodology.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import dspy


EVALUATION_PROMPT = """\
You are grading a model's answer to a question that required retrieving information from web pages.

Grading rules:
- The answer is CORRECT if it contains the key fact(s) from the reference answer, even if phrased differently.
- For named entities (people, places, teams), the name must match exactly (minor spelling variants are fine).
- For numbers and dates, the value must match exactly.
- If the model's answer is empty or clearly a non-answer, score it 0.

Question: {question}
Model Answer: {model_answer}
Reference Answer: {reference_answer}

Score the model answer with 0 (wrong) or 1 (correct).
Correctness:"""


def _parse_score(response: str) -> int:
    match = re.search(r"\b([01])\b", response)
    if match:
        return int(match.group(1))
    return 1 if "1" in response else 0


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace_path = Path(workspace_dir)
    example_id = example["query_id"]
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
        raise ValueError("browsecomplongcontext evaluation requires an LM judge (eval_lm must be set)")

    prompt = EVALUATION_PROMPT.format(
        question=question,
        model_answer=agent_answer,
        reference_answer=reference_answer,
    )

    response = lm(messages=[{"role": "user", "content": prompt}])
    response_text = response[0] if isinstance(response, list) else str(response)

    score = float(_parse_score(response_text))

    return {
        "workspace_dir": workspace_dir,
        "example_id": example_id,
        "score": score,
        "feedback": f"Correctness: {int(score)} (raw: {response_text.strip()!r})\nAgent answer: {agent_answer}\nReference answer: {reference_answer}",
        "agent_answer": agent_answer,
        "reference_answer": reference_answer,
    }
