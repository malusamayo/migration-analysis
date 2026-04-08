"""Evaluation for the replicatorbench task.

Four-stage LLM-as-judge evaluation mirroring the original ReplicatorBench protocol:

  1. Extraction  — post_registration.json vs expected_post_registration.json
  2. Design      — replication_info.json vs human_preregistration.pdf
  3. Execution   — execution_results.json scored against a structured rubric
  4. Interpretation — interpret_results.json vs human_report.pdf

Each stage uses the rubric prompts from the original benchmark, adapted for dspy.
Overall score is the unweighted mean of the four stage scores.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy

from ..utils import batch_inference


# ---------------------------------------------------------------------------
# Embedded rubric prompt templates (verbatim from original benchmark)
# ---------------------------------------------------------------------------

_EXTRACT_EVAL_PROMPT = """
You are an information verifier. The task you're concerned with is extracting important information about a main claim in a research paper.
You are given TWO json objects: one extracted and one reference, your task is to score the information (key, value pair) presented in the extraced JSON object based on the (key, value pair) presented in the reference JSON object.

=== START OF EXPLANATION OF WHAT EACH FIELD IN THE JSON MEAN ===
{extraction_schema}
=== END OF EXPLANATION OF WHAT EACH FIELD IN THE JSON MEAN ===

Follow the rubrics below for your evaluation of each component in the JSON. The rubric uses a 0-3 scoring scale for all components, where:
- 3: Exact Match - The extracted information is identical or nearly identical to the reference in the paper. The meaning is preserved completely, with the same level of detail, including all key elements (e.g., variables, relationships, or numerical values). No omissions or additions of unrelated information.
- 2: Mostly Similar - The extracted information conveys the same core meaning as the reference, but with different phrasing or structure. Minor details may be omitted, but the essential information is preserved. If the extracted content is a finding that directly supports a hypothesis, consider it equivalent.
- 1: Loosely Related - The extracted information has partial overlap with the reference but includes significant differences, omissions of major details, or additions of unrelated information. The core meaning is somewhat preserved but incomplete or altered.
- 0: No Match - No relevant overlap with the reference, completely incorrect, or missing entirely.

For fields where the reference is "not stated" or "NA", assign a score of 3 as long as the extracted information reflect similar meaning: "not available", "not stated", etc.

=== EXTRACTED JSON TO BE EVALUATED START ===
{extracted_json}
=== EXTRACTED JSON TO BE EVALUATED END ===

=== REFERENCE JSON WITH THE CORRECT INFO ===
{expected_json}
=== REFERENCE JSON WITH THE CORRECT INFO ===

Please return your evaluation as a JSON object where each key is a specific component from the original JSON. For example:
{{
    "claim.hypothesis": {{
    "score": 3,
    "explanation": "reasoning for your scoring."
    }},
    "results.numerical_results[0].outcome_name": {{
    "score": 2,
    "explanation": "reasoning for your scoring."
    }},
    ...
}}
Return a valid JSON object only. Do NOT wrap the output in markdown. Do NOT include extra text or commentary.
""".strip()

_DESIGN_EVAL_PROMPT = """
You are an information verifier.
You are given a json object and a reference document, your task is to score the information (key, value pair) presented in the extracted JSON object based on the information presented in the reference document.

=== START OF EXPLANATION OF WHAT EACH FIELD IN THE JSON MEAN ===
{extraction_schema}
=== END OF EXPLANATION OF WHAT EACH FIELD IN THE JSON MEAN ===

Follow the rubrics below for your evaluation of each component in the JSON. The rubric uses a 0-3 scoring scale for all components, where:
- 3: Exact Match - The extracted information is identical or nearly identical to the reference in the document. The meaning is preserved completely, with the same level of detail, including all key elements (e.g., variables, relationships, or numerical values). No omissions or additions of unrelated information.
- 2: Mostly Similar - The extracted information conveys the same core meaning as the reference, but with different phrasing or structure. Minor details may be omitted, but the essential information is preserved. If the extracted content is a finding that directly supports a hypothesis, consider it equivalent.
- 1: Loosely Related - The extracted information has partial overlap with the reference but includes significant differences, omissions of major details, or additions of unrelated information. The core meaning is somewhat preserved but incomplete or altered.
- 0: No Match - No relevant overlap with the reference, completely incorrect, or missing entirely.
- NA: The reference document does not contain information to evaluate this component/field of the JSON.

=== START OF EXTRACTED JSON TO BE EVALUATED START ===
{extracted_json}
=== END OF EXTRACTED JSON TO BE EVALUATED END ===

=== START OF REFERENCE DOCUMENT WITH THE CORRECT INFO ===
{reference_doc}
=== END OF REFERENCE DOCUMENT WITH THE CORRECT INFO ===

Please return your evaluation as a JSON object where each key is a specific component from the original JSON. For example:
{{
    "hypothesis": {{
    "score": 3,
    "explanation": "reasoning for your scoring."
    }},
    "data_plan.source_type": {{
    "score": 2,
    "explanation": "reasoning for your scoring."
    }},
    ...
}}
Return a valid JSON object only. Do NOT wrap the output in markdown. Do NOT include extra text or commentary.
""".strip()

_EXECUTE_EVAL_PROMPT = """
You are an experienced researcher. You are assessing a replication attempt of a research claim.

You are given the following information about the research claim and original study:
{claim_context}

You are given the following information about the replication attempt:
{agent_context}

Your task is to score the replication attempt by filling out this structured rubric:
=== START OF EVALUATION RUBRIC TO BE FILLED OUT ===
{rubric_schema}
=== END OF EVALUATION RUBRIC TO BE FILLED OUT ===

For each leaf node in the rubric assign a score of 0 or 1 based on the rubric requirement, with 1 being a satisfaction of the rubric requirement.

Please return your evaluation as a JSON object matching the rubric structure, where each leaf node has a "score" (0 or 1) and "explanation". For example:
{{
    "evaluate_design.environment.1.1.1": {{
    "score": 1,
    "explanation": "reasoning."
    }},
    ...
}}
Return a valid JSON object only. Do NOT wrap the output in markdown. Do NOT include extra text or commentary.
""".strip()

_INTERPRET_EVAL_PROMPT = """
You are a researcher specialized in evaluating research replication studies.
You are given a JSON object containing a structured report of a replication attempt of a research paper and a reference document that contains outcomes/information if the replication study is carried out correctly. Your task is to score the information (key, value pair) presented in the reported JSON object based on the information presented in the reference document.

=== START OF EXPLANATION OF WHAT EACH FIELD IN THE JSON MEAN ===
{interpret_schema}
=== END OF EXPLANATION OF WHAT EACH FIELD IN THE JSON MEAN ===

Follow the rubrics below for your evaluation of each component in the "fidelity_assessment" and "results_comparison" aspects:
- 3: Exact Match - The reported information is identical or nearly identical to the reference in the document.
- 2: Mostly Similar - The reported information conveys the same core meaning as the reference, but with different phrasing or structure.
- 1: Loosely Related - The reported information has partial overlap with the reference but includes significant differences.
- 0: No Match - No relevant overlap with the reference, completely incorrect, or missing entirely.
- NA: The reference document does not contain information to evaluate this component.

For "interpretation_summary" and "execute_status": an integer 0-3 based on quality and completeness.
For "failure_handling" and "notes": an integer 0-3 for clarity, specificity, and feasibility.

=== START OF EXTRACTED JSON TO BE EVALUATED ===
{reported_json}
=== END OF EXTRACTED JSON TO BE EVALUATED ===

=== START OF REFERENCE DOCUMENT WITH THE CORRECT INFO ===
{reference_report_doc}
=== END OF REFERENCE DOCUMENT WITH THE CORRECT INFO ===

Please return your evaluation as a JSON object where each key is a leaf field from the original JSON. For example:
{{
    "interpretation_summary": {{
    "score": 2,
    "explanation": "reasoning."
    }},
    "results_comparison.overall_answer": {{
    "score": 3,
    "explanation": "reasoning."
    }},
    ...
}}
Return a valid JSON object only. Do NOT wrap the output in markdown. Do NOT include extra text or commentary.
""".strip()

# ---------------------------------------------------------------------------
# Embedded schemas (for field-explanation context in prompts)
# ---------------------------------------------------------------------------

_POST_REGISTRATION_SCHEMA = """{
  "original_study.claim.hypotheses": "Testable hypothesis based on the claim",
  "original_study.claim.hypotheses_location": "Where the hypothesis appears in the paper",
  "original_study.claim.statement": "The main claim made by the original study",
  "original_study.claim.statement_location": "Where the claim is stated in the paper",
  "original_study.claim.study_type": "Type of study (Experimental, Observational, Meta-Analysis)",
  "original_study.data.source": "Data source (e.g., survey, database)",
  "original_study.data.wave_or_subset": "Specific waves or subsets if applicable",
  "original_study.data.sample_size": "Sample size of the selected data",
  "original_study.data.unit_of_analysis": "Unit of analysis (e.g., individual, household)",
  "original_study.data.access_details": "Access restrictions or request process",
  "original_study.data.notes": "Additional caveats (encoding, nested structure, etc.)",
  "original_study.method.description": "Narrative summary of how the study was conducted",
  "original_study.method.steps": "Ordered procedural steps to reproduce the study",
  "original_study.method.models": "Statistical model or approach",
  "original_study.method.outcome_variable": "Dependent variable",
  "original_study.method.independent_variables": "Primary predictors",
  "original_study.method.control_variables": "Variables controlled for",
  "original_study.method.tools_software": "Software or packages mentioned",
  "original_study.results.summary": "Narrative summary of main findings",
  "original_study.results.numerical_results[].outcome_name": "Label for this result",
  "original_study.results.numerical_results[].value": "Numeric result value",
  "original_study.results.numerical_results[].confidence_interval": "CI bounds and level",
  "original_study.results.numerical_results[].p_value": "P-value",
  "original_study.results.numerical_results[].statistical_significance": "Boolean",
  "original_study.results.numerical_results[].direction": "positive, negative, or null",
  "original_study.metadata.original_paper_id": "DOI or identifier",
  "original_study.metadata.original_paper_title": "Full title"
}"""

_PRE_REGISTRATION_SCHEMA = """{
  "replication_study.hypothesis": "Focal hypothesis phrased as a testable statement",
  "replication_study.study_type": "Study type for the replication",
  "replication_study.data_plan.dataset_identifier": "Name/version of dataset",
  "replication_study.data_plan.source_type": "Data source type",
  "replication_study.data_plan.wave_or_subset": "Waves or subsets to use",
  "replication_study.data_plan.sample_size": "Expected sample size",
  "replication_study.data_plan.unit_of_analysis": "Unit of analysis",
  "replication_study.data_plan.qualification.explanation": "Why this dataset is appropriate",
  "replication_study.data_plan.qualification.similarity_to_original": "How it matches the original",
  "replication_study.data_plan.qualification.deviation_from_original": "How it differs from original",
  "replication_study.planned_method.steps": "Ordered procedural steps",
  "replication_study.planned_method.models": "Statistical model to use",
  "replication_study.planned_method.outcome_variable": "Dependent variable",
  "replication_study.planned_method.independent_variables": "Primary predictors",
  "replication_study.planned_method.control_variables": "Variables to control for",
  "replication_study.planned_method.inference_criteria": "Rules for judging support"
}"""

_INTERPRET_SCHEMA = """{
  "interpretation_summary": "Narrative overview of the assessment",
  "execute_status": "Overall execution status (Success, Partial Success, Failure)",
  "fidelity_assessment.method_alignment": "How well executed methods matched the preregistration",
  "fidelity_assessment.deviations": "List of deviations with impact assessments",
  "results_comparison.hypothesis_tested": "Restatement of the focal hypothesis",
  "results_comparison.original_results": "Summary of original findings with numerical values",
  "results_comparison.replication_results": "Summary of replication findings",
  "results_comparison.overall_answer": "Do replication results satisfy preregistered criteria?",
  "replication_report": "Short summary of overall outcome",
  "failure_handling": "List of failures with actionable suggestions",
  "notes": "Additional caveats or suggestions"
}"""

_EXECUTE_RUBRIC = """{
  "evaluate_design": {
    "environment": {
      "1.1.1": "Verify that docker_specs.base_image exists in replication_info.json",
      "1.1.2": "Check for missing manifest or dependency declarations"
    },
    "dependency": {
      "1.2": "Agent successfully identified imports/libraries and reported them under docker_specs.packages"
    },
    "file_system": {
      "1.3.1": "Agent detected and handled hard-coded paths",
      "1.3.2": "replication_info.codebase.files entries exist in replication_data folder",
      "1.3.3": "If data is to be mounted, it has a correct path specification"
    }
  },
  "execute": {
    "code_execution": {
      "2.1.1": "Data is successfully loaded (evidenced by execution_results)",
      "2.2.2": "Main code or model is executed without errors"
    },
    "execution_report": {
      "2.3.1": "Expected output files generated by code are logged and reported",
      "2.3.2": "execution_results.json report is filled out with actual findings"
    }
  }
}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pdf_to_text(path: Path) -> str:
    import fitz  # pymupdf
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def _read_json(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    if not path.exists():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.rstrip())
    return text.strip()

def _normalize_lm_json(text: str) -> str:
    """Fix common LLM JSON quirks before parsing."""
    # Replace bare NA (not inside quotes) with "NA"
    text = re.sub(r':\s*NA\b', ': "NA"', text)
    return text


def _call_lm_judge(lm: dspy.LM, prompt: str, retries: int = 2) -> Tuple[Optional[dict], Optional[str]]:
    last_err = "no attempts made"
    for attempt in range(retries):
        response = lm(messages=[{"role": "user", "content": prompt}])
        text = response[0] if isinstance(response, list) else str(response)
        text = _strip_fences(text)
        text = _normalize_lm_json(text)
        # First try parsing the whole response
        try:
            return json.loads(text), None
        except json.JSONDecodeError as exc:
            last_err = f"LLM returned malformed JSON: {exc}\nRaw: {text}"
    return None, last_err


def _get_nested_value(data: dict, path: str) -> Any:
    """Navigate a dotted/bracketed field path in a nested dict/list."""
    keys = re.split(r'[.\[]', path)
    keys = [k.rstrip(']') for k in keys if k]
    current = data
    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return None
            current = current[key]
        elif isinstance(current, list):
            try:
                current = current[int(key)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _iter_eval_entries(result: dict, prefix: str = "") -> List[Tuple[str, dict]]:
    entries: List[Tuple[str, dict]] = []
    for field, entry in result.items():
        field_path = f"{prefix}.{field}" if prefix else str(field)
        if not isinstance(entry, dict):
            continue
        if "score" in entry:
            entries.append((field_path, entry))
            continue
        entries.extend(_iter_eval_entries(entry, prefix=field_path))
    return entries


def _score_eval_dict(result: dict, max_score: float = 3.0, raw_data: Optional[dict] = None) -> Tuple[float, List[str]]:
    """Normalize an eval dict of {field: {score, explanation}} to 0–1."""
    scores: List[float] = []
    lines: List[str] = []
    for field, entry in _iter_eval_entries(result):
        raw = entry.get("score")
        explanation = entry.get("explanation", "")
        if raw == "NA" or raw is None:
            continue
        try:
            score_f = float(raw)
        except (ValueError, TypeError):
            continue
        normalized = score_f / max_score
        scores.append(normalized)
        if score_f >= max_score:
            mark = "PASS"
        elif score_f > 0:
            mark = f"PART {score_f:.0f}/{max_score:.0f}"
        else:
            mark = "FAIL"
        # raw_val = _get_nested_value(raw_data, field) if raw_data is not None else None
        # raw_str = f" | answer: {json.dumps(raw_val, ensure_ascii=False)}" if raw_val is not None else ""
        lines.append(f"  [{mark}] {field}: {explanation}")
    if not scores:
        return 0.0, lines
    return sum(scores) / len(scores), lines


# ---------------------------------------------------------------------------
# Stage scorers
# ---------------------------------------------------------------------------

def _score_extraction(
    workspace: Path,
    gt_dir: Path,
    lm: dspy.LM,
) -> Tuple[float, str]:
    agent_data, err = _read_json(workspace / "post_registration.json")
    if agent_data is None:
        return 0.0, f"extraction: post_registration.json {err}"

    gt_path = gt_dir / "expected_post_registration.json"
    if not gt_path.exists():
        return 0.0, "extraction: expected_post_registration.json not found in groundtruth"
    try:
        expected = json.loads(gt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return 0.0, f"extraction: expected_post_registration.json invalid json: {exc}"

    prompt = _EXTRACT_EVAL_PROMPT.format(
        extraction_schema=_POST_REGISTRATION_SCHEMA,
        extracted_json=json.dumps(agent_data, indent=2),
        expected_json=json.dumps(expected, indent=2),
    )
    result, err = _call_lm_judge(lm, prompt)
    if result is None:
        return 0.0, f"extraction: {err}"

    score, lines = _score_eval_dict(result, max_score=3.0, raw_data=agent_data)
    feedback_lines = [f"extraction: {score:.3f}"] + lines
    return score, "\n".join(feedback_lines)


def _score_design(
    workspace: Path,
    gt_dir: Path,
    lm: dspy.LM,
) -> Tuple[float, str]:
    agent_data, err = _read_json(workspace / "replication_info.json")
    if agent_data is None:
        return 0.0, f"design: replication_info.json {err}"

    preregistration_pdf = gt_dir / "human_preregistration.pdf"
    if not preregistration_pdf.exists():
        return 0.0, "design: human_preregistration.pdf not found in groundtruth"

    reference_doc = _pdf_to_text(preregistration_pdf)
    prompt = _DESIGN_EVAL_PROMPT.format(
        extraction_schema=_PRE_REGISTRATION_SCHEMA,
        extracted_json=json.dumps(agent_data, indent=2),
        reference_doc=reference_doc,
    )
    result, err = _call_lm_judge(lm, prompt)
    if result is None:
        return 0.0, f"design: {err}"

    score, lines = _score_eval_dict(result, max_score=3.0, raw_data=agent_data)
    feedback_lines = [f"design: {score:.3f}"] + lines
    return score, "\n".join(feedback_lines)


def _score_execution(
    workspace: Path,
    lm: dspy.LM,
) -> Tuple[float, str]:
    exec_data, err = _read_json(workspace / "execution_results.json")
    if exec_data is None:
        return 0.0, f"execution: execution_results.json {err}"

    # Build context from available agent files
    claim_parts = []
    post_reg, _ = _read_json(workspace / "post_registration.json")
    if post_reg:
        claim_parts.append(f"post_registration.json:\n{json.dumps(post_reg, indent=2)}")

    agent_parts = [f"execution_results.json:\n{json.dumps(exec_data, indent=2)}"]
    repl_info, _ = _read_json(workspace / "replication_info.json")
    if repl_info:
        agent_parts.append(f"replication_info.json:\n{json.dumps(repl_info, indent=2)}")

    prompt = _EXECUTE_EVAL_PROMPT.format(
        claim_context="\n\n".join(claim_parts) if claim_parts else "(not available)",
        agent_context="\n\n".join(agent_parts),
        rubric_schema=_EXECUTE_RUBRIC,
    )
    result, err = _call_lm_judge(lm, prompt)
    if result is None:
        return 0.0, f"execution: {err}"

    score, lines = _score_eval_dict(result, max_score=1.0)
    feedback_lines = [f"execution: {score:.3f}"] + lines
    return score, "\n".join(feedback_lines)


def _score_interpretation(
    workspace: Path,
    gt_dir: Path,
    lm: dspy.LM,
) -> Tuple[float, str]:
    agent_data, err = _read_json(workspace / "interpret_results.json")
    if agent_data is None:
        return 0.0, f"interpretation: interpret_results.json {err}"

    report_pdf = gt_dir / "human_report.pdf"
    if not report_pdf.exists():
        return 0.0, "interpretation: human_report.pdf not found in groundtruth"

    reference_doc = _pdf_to_text(report_pdf)
    prompt = _INTERPRET_EVAL_PROMPT.format(
        interpret_schema=_INTERPRET_SCHEMA,
        reported_json=json.dumps(agent_data, indent=2),
        reference_report_doc=reference_doc,
    )
    result, err = _call_lm_judge(lm, prompt)
    if result is None:
        return 0.0, f"interpretation: {err}"

    score, lines = _score_eval_dict(result, max_score=3.0, raw_data=agent_data)
    feedback_lines = [f"interpretation: {score:.3f}"] + lines
    return score, "\n".join(feedback_lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    if lm is None:
        raise ValueError("replicatorbench evaluation requires an LM judge (eval_lm must be set)")

    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    gt_dir = log_dir / "groundtruth"

    if not gt_dir.exists():
        return {
            "workspace_dir": workspace_dir,
            "example_id": example.get("id"),
            "score": 0.0,
            "feedback": "Ground truth directory not found. Workspace setup may have failed.",
        }

    def _run_stage(stage_fn, **kwargs):
        return stage_fn(**kwargs)

    stage_args = [
        {"stage_fn": _score_extraction, "workspace": workspace, "gt_dir": gt_dir, "lm": lm},
        {"stage_fn": _score_design,     "workspace": workspace, "gt_dir": gt_dir, "lm": lm},
        {"stage_fn": _score_execution,  "workspace": workspace,                   "lm": lm},
        {"stage_fn": _score_interpretation, "workspace": workspace, "gt_dir": gt_dir, "lm": lm},
    ]
    stage_results = batch_inference(_run_stage, stage_args, max_workers=4)
    (extraction_score, extraction_feedback) = stage_results[0]
    (design_score,     design_feedback)     = stage_results[1]
    (execution_score,  execution_feedback)  = stage_results[2]
    (interpretation_score, interpretation_feedback) = stage_results[3]

    overall = (extraction_score + design_score + execution_score + interpretation_score) / 4.0

    feedback = "\n\n".join([
        f"Overall score: {overall:.3f}",
        extraction_feedback,
        design_feedback,
        execution_feedback,
        interpretation_feedback,
    ])

    return {
        "workspace_dir": workspace_dir,
        "example_id": example.get("id"),
        "score": overall,
        "stage_scores": {
            "extraction": extraction_score,
            "design": design_score,
            "execution": execution_score,
            "interpretation": interpretation_score,
        },
        "feedback": feedback,
    }
