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


class WebarenaFeedbackSignature(dspy.Signature):
    """Generate brief feedback for a webarena task evaluation result."""
    intent = dspy.InputField(desc="The web navigation task the agent was asked to complete.")
    raw_eval_output = dspy.InputField(desc="The raw final output from the agent (may be a JSON string from the FinishTool, plain text, or empty).")
    agent_answer = dspy.InputField(desc="The answer extracted from the agent's raw output. Empty string means extraction failed (agent did not produce a parseable structured answer).")
    reference_answers = dspy.InputField(desc="The correct reference answer(s) as a JSON string.")
    score = dspy.InputField(desc="Evaluation score from 0.0 (wrong) to 1.0 (correct).")
    feedback = dspy.OutputField(desc="One or two sentences: state if correct, or explain the mismatch. If agent_answer is empty, note that the agent failed to produce a structured answer via FinishTool. Otherwise explain what the agent got wrong vs the reference.")


def _build_feedback(
    lm: Optional[dspy.LM],
    intent: str,
    raw_eval_output: str,
    agent_answer: str,
    reference_answers: dict,
    score: float,
) -> str:
    if score >= 1.0:
        return "Correct."
    ref_str = json.dumps(reference_answers)
    if lm is None:
        if not agent_answer:
            return f"Incorrect. Agent did not produce a structured answer (raw output: {repr(raw_eval_output[:200])}). Expected {ref_str}."
        return f"Incorrect. Expected {ref_str}, got: {repr(agent_answer[:200])}"
    with dspy.context(lm=lm):
        result = dspy.Predict(WebarenaFeedbackSignature)(
            intent=intent,
            raw_eval_output=raw_eval_output[:500],
            agent_answer=agent_answer[:500],
            reference_answers=ref_str,
            score=str(score),
        )
    return result.feedback


def _extract_from_json(data: dict) -> Optional[str]:
    """Extract an answer string from a parsed JSON dict.

    Checks keys in priority order; skips empty lists/strings.
    For list values, joins string items with ', ' and extracts 'name' from dict items.
    error_details is checked last to support unachievable-task answers.
    """
    for key in ("retrieved_data", "answer", "result", "value", "error_details"):
        val = data.get(key)
        if not val and val != 0:  # skip None, [], ""
            continue
        if isinstance(val, list):
            parts = []
            for item in val:
                if isinstance(item, dict):
                    parts.append(item.get("name", item.get("title", str(item))))
                else:
                    parts.append(str(item))
            return ", ".join(parts)
        else:
            return str(val)
    return None


def _extract_answer(eval_output: str) -> str:
    """Extract the answer from the agent's output.

    Tries in order:
    1. Explicit ``ANSWER: <text>`` prefix (original format)
    2. Fenced JSON block (``` json {...} ```)
    3. Full string is a JSON object
    4. JSON object embedded anywhere in the text
    Returns "" if no structured answer is found (avoids full-output false positives).
    """
    # 1. ANSWER: prefix
    match = re.search(r"^ANSWER:\s*(.+)", eval_output, re.MULTILINE | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2. Fenced JSON block
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", eval_output, re.DOTALL)
    if json_match:
        try:
            extracted = _extract_from_json(json.loads(json_match.group(1)))
            if extracted is not None:
                return extracted
        except (json.JSONDecodeError, AttributeError):
            pass

    # 3. Full string is a JSON object (FinishTool emits {"task_type": ..., "retrieved_data": [...]} directly)
    try:
        data = json.loads(eval_output.strip())
        if isinstance(data, dict):
            extracted = _extract_from_json(data)
            if extracted is not None:
                return extracted
    except (json.JSONDecodeError, ValueError):
        pass

    # 4. JSON object embedded somewhere in the text
    json_match = re.search(r'\{.*?\}', eval_output, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if isinstance(data, dict):
                extracted = _extract_from_json(data)
                if extracted is not None:
                    return extracted
        except (json.JSONDecodeError, ValueError):
            pass

    # No structured answer found — return empty string so eval scores 0, not a false match
    return ""


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
            "feedback": "No trace file found — agent did not run.",
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
        reference_answers = eval_config.get("reference_answers", {})
        answer = _extract_answer(eval_output)
        score = _string_match_score(
            pred=answer,
            reference_answers=reference_answers,
            intent=example.get("prompt", ""),
            string_note=eval_config.get("string_note", ""),
            lm=lm,
        )
        result["score"] = score
        result['raw_output'] = eval_output
        result["answer"] = answer
        result["reference_answers"] = reference_answers
        result["feedback"] = _build_feedback(
            lm=lm,
            intent=example.get("prompt", ""),
            raw_eval_output=eval_output,
            agent_answer=answer,
            reference_answers=reference_answers,
            score=score,
        )
    else:
        # url_match / program_html require live browser
        result["score"] = None
        result["note"] = f"eval_types={eval_types} require live browser (not evaluated offline)"
        result["agent_responded"] = bool(
            re.search(r"^(ANSWER|COMPLETED):", eval_output, re.MULTILINE | re.IGNORECASE)
        )

    return result
