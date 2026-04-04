"""Evaluation functions for docbench task.

Agents read a PDF document and write their answer to answer.txt.
Evaluation uses an LLM judge following the original DocBench evaluation prompt
from https://github.com/Anni-Zou/DocBench/blob/main/evaluation_prompt.txt
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

import dspy


EVALUATION_PROMPT = """\
Task Overview:
You are tasked with evaluating user answers based on a given question, reference answer, and additional reference text. Your goal is to assess the correctness of the user answer using a specific metric.

Evaluation Criteria:
1. Yes/No Questions: Verify if the user's answer aligns with the reference answer in terms of a "yes" or "no" response.
2. Short Answers/Directives: Ensure key details such as numbers, specific nouns/verbs, and dates match those in the reference answer.
3. Abstractive/Long Answers: The user's answer can differ in wording but must convey the same meaning and contain the same key information as the reference answer to be considered correct.

Evaluation Process:
1. Identify the type of question presented.
2. Apply the relevant criteria from the Evaluation Criteria.
3. Compare the user's answer against the reference answer accordingly.
4. Consult the reference text for clarification when needed.
5. Score the answer with a binary label 0 or 1, where 0 denotes wrong and 1 denotes correct.
NOTE that if the user answer is 0 or an empty string, it should get a 0 score.

Question: {question}
User Answer: {sys_ans}
Reference Answer: {ref_ans}
Reference Text: {ref_text}

Evaluation Form (score ONLY):
- Correctness:"""


def _parse_correctness(response: str) -> int:
    """Extract the binary correctness score (0 or 1) from LM response."""
    # Look for the first occurrence of 0 or 1 in the response
    match = re.search(r"\b([01])\b", response)
    if match:
        return int(match.group(1))
    # Fallback: if "1" appears anywhere treat as correct, else 0
    return 1 if "1" in response else 0


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace_path = Path(workspace_dir)
    example_id = example.get("id", example.get("task_id"))
    reference_answer = example["answer"]
    question = example["question"]
    evidence = example.get("evidence", "")

    answer_file = workspace_path / "answer.txt"
    if not answer_file.exists():
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "feedback": "answer.txt not found in workspace.",
            "agent_answer": "",
            "reference_answer": reference_answer,
            "question_type": example.get("question_type", ""),
        }

    agent_answer = answer_file.read_text(encoding="utf-8").strip()

    if not agent_answer or agent_answer == "0":
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "feedback": "Answer is empty or '0'.",
            "agent_answer": agent_answer,
            "reference_answer": reference_answer,
            "question_type": example.get("question_type", ""),
        }

    if lm is None:
        raise ValueError("docbench evaluation requires an LM judge (eval_lm must be set)")

    prompt = EVALUATION_PROMPT.format(
        question=question,
        sys_ans=agent_answer,
        ref_ans=reference_answer,
        ref_text=evidence,
    )

    response = lm(messages=[{"role": "user", "content": prompt}])
    # dspy LM returns a list of completion strings
    response_text = response[0] if isinstance(response, list) else str(response)

    correctness = _parse_correctness(response_text)
    score = float(correctness)

    return {
        "workspace_dir": workspace_dir,
        "example_id": example_id,
        "score": score,
        "feedback": f"Correctness: {correctness} (raw: {response_text.strip()!r})",
        "agent_answer": agent_answer,
        "reference_answer": reference_answer,
        "question_type": example.get("question_type", ""),
    }
