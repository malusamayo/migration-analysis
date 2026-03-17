"""Evaluation functions for webarena task.

string_match tasks are evaluated offline by comparing the extracted ANSWER
against the reference answers embedded in the task config.

url_match and program_html tasks require a live browser; they return score=None.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import dspy


def _extract_answer(eval_output: str) -> str:
    """Extract the answer from the agent's output.

    Tries in order:
    1. Explicit ``ANSWER: <text>`` prefix (original format)
    2. ``retrieved_data`` field inside a fenced JSON block
    3. Full output text (allows must_include / fuzzy_match to work on natural-language answers)
    """
    # 1. ANSWER: prefix
    match = re.search(r"^ANSWER:\s*(.+)", eval_output, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. JSON block with retrieved_data / answer field
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", eval_output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            for key in ("retrieved_data", "answer", "result", "value"):
                val = data.get(key)
                if val:
                    if isinstance(val, list):
                        return ", ".join(str(x) for x in val)
                    return str(val)
        except (json.JSONDecodeError, AttributeError):
            pass

    # 3. Fall back to full output (supports must_include / fuzzy_match on natural-language answers)
    return eval_output


def _clean_answer(answer: str) -> str:
    answer = answer.strip()
    if answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    elif answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    return answer.lower()


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer matching nltk word_tokenize behavior for alphanumeric tokens."""
    return re.findall(r'\b\w+\b', text.lower())


def _must_include(ref: str, pred: str, tokenize: bool = False) -> float:
    clean_ref = _clean_answer(ref)
    clean_pred = _clean_answer(pred)
    # Replicate official: tokenize only when there's 1 must_include item and ref is a single char.
    # This prevents false positives like "0" in "100" → True.
    if tokenize and len(clean_ref) == 1:
        return float(clean_ref in _tokenize(clean_pred))
    return float(clean_ref in clean_pred)


def _llm_fuzzy_match(lm: dspy.LM, pred: str, ref: str, intent: str) -> float:
    prompt = (
        f"Your task is to evaluate whether a model response is equivalent to a reference answer "
        f"for the following web navigation task.\n"
        f"Task: {intent}\n"
        f"Reference answer: {ref}\n"
        f"Model response: {pred}\n\n"
        f"Does the model response match the reference answer? "
        f"Respond with 'yes' or 'no' only."
    )
    response = lm(messages=[{"role": "user", "content": prompt}])
    return float(response[0].strip().lower().startswith("yes"))


def _llm_ua_match(lm: dspy.LM, pred: str, ref: str, intent: str) -> float:
    """Judge whether the agent correctly identified an unachievable task."""
    prompt = (
        f"Your task is to evaluate whether a model correctly identifies that the following "
        f"web navigation task is unachievable.\n"
        f"Task: {intent}\n"
        f"Expected reason the task is unachievable: {ref}\n"
        f"Model response: {pred}\n\n"
        f"Did the model correctly identify that the task is unachievable and provide "
        f"an appropriate reason? Respond with 'yes' or 'no' only."
    )
    response = lm(messages=[{"role": "user", "content": prompt}])
    return float(response[0].strip().lower().startswith("yes"))


def _string_match_score(
    pred: str,
    reference_answers: dict,
    intent: str = "",
    string_note: str = "",
    lm: Optional[dspy.LM] = None,
) -> float:
    pred_clean = _clean_answer(pred)
    score = 1.0
    for approach, value in reference_answers.items():
        if approach == "exact_match":
            score *= float(_clean_answer(str(value)) == pred_clean)
        elif approach == "must_include":
            assert isinstance(value, list)
            for must_value in value:
                score *= _must_include(
                    ref=must_value,
                    pred=pred,
                    tokenize=(len(value) == 1),
                )
        elif approach == "fuzzy_match":
            if value == "N/A":
                # Replicate official: exact match first, then ua_match fallback
                score *= float(_clean_answer("N/A") == pred_clean)
                if score != 1.0 and lm is not None:
                    # Official replaces accumulated score (not multiply) on ua_match
                    score = 1.0 * _llm_ua_match(lm, pred=pred_clean, ref=string_note, intent=intent)
            else:
                assert isinstance(value, list)
                for reference in value:
                    if lm is not None:
                        score *= _llm_fuzzy_match(lm, pred=pred_clean, ref=reference, intent=intent)
                    # If no lm provided, skip fuzzy_match (treat as passing)
    return score


def run_single_instance_eval(
    lm: dspy.LM,
    workspace_dir: str,
    example: Optional[dict] = None,
) -> Dict[str, Any]:
    workspace_path = Path(workspace_dir)
    log_dir = workspace_path.parent / f"{workspace_path.name}_logs"

    trace_files = list(log_dir.glob("trace_*.json"))
    if not trace_files:
        return {
            "workspace_dir": str(workspace_dir),
            "score": 0.0,
            "error": "No trace file found",
        }

    with open(trace_files[0]) as f:
        trace = json.load(f)

    eval_output = trace.get("eval_output", "")
    task_id = int(example["task_id"])
    eval_config = example["eval"]
    eval_types = eval_config["eval_types"]

    result: Dict[str, Any] = {
        "workspace_dir": str(workspace_dir),
        "task_id": task_id,
        "eval_types": eval_types,
    }

    if eval_types == ["string_match"]:
        answer = _extract_answer(eval_output)
        result["score"] = _string_match_score(
            pred=answer,
            reference_answers=eval_config.get("reference_answers", {}),
            intent=example.get("prompt", ""),
            string_note=eval_config.get("string_note", ""),
            lm=lm,
        )
        result["answer"] = answer[:500] if len(answer) > 500 else answer
    else:
        # url_match / program_html require live browser
        result["score"] = None
        result["note"] = f"eval_types={eval_types} require live browser (not evaluated offline)"
        result["agent_responded"] = bool(
            re.search(r"^(ANSWER|COMPLETED):", eval_output, re.MULTILINE | re.IGNORECASE)
        )

    return result
