"""Evaluation for the replicatorbench task.

Scores post_registration.json against ground truth on deterministic numeric
and categorical fields only — no LLM judge.

Graded fields (all sourced from original_study):
  - data.sample_size            exact integer match (after stripping commas/prose)
  - results.numerical_results[] per result in the reference, find the best-matching
                                 agent result and check:
                                   value              2% relative tolerance
                                   confidence_interval.lower/upper   2% rel tol
                                   direction          exact (case-insensitive)
                                   statistical_significance  boolean match

Fields annotated "not stated" in the ground truth are skipped.
Score = mean over all graded fields across all numerical results.
If both expected_post_registration.json and expected_post_registration_2.json
are present, the higher score is kept.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dspy


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_float(v) -> Optional[float]:
    """Extract a float from a value (number or string like '4.12', '<0.001')."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"[\d]+\.?[\d]*(?:e[+-]?\d+)?", str(v))
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def _parse_int(v) -> Optional[int]:
    """Extract the first integer from a string or number (handles '3,037 counties')."""
    if v is None:
        return None
    if isinstance(v, int):
        return v
    m = re.search(r"[\d,]+", str(v))
    if m:
        try:
            return int(m.group(0).replace(",", ""))
        except ValueError:
            return None
    return None


def _parse_bool(v) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    s = str(v).lower().strip()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    return None


def _is_not_stated(v) -> bool:
    """True if the value represents a missing/unstated field in the ground truth."""
    if v is None:
        return True
    return str(v).lower().strip() in ("not stated", "na", "n/a", "")


def _floats_close(a, b, rel_tol: float = 0.02) -> bool:
    """True if the two values are within rel_tol relative tolerance."""
    fa, fb = _parse_float(a), _parse_float(b)
    if fa is None or fb is None:
        return False
    if fa == 0.0 and fb == 0.0:
        return True
    return abs(fa - fb) / max(abs(fa), abs(fb)) < rel_tol


# ---------------------------------------------------------------------------
# Scoring logic — returns (score, lines) so callers can build feedback
# ---------------------------------------------------------------------------

# A check record: (field_label, passed, expected_repr, got_repr)
_Check = Tuple[str, bool, str, str]


def _check_numerical_result(agent: dict, ref: dict, prefix: str) -> List[_Check]:
    checks: List[_Check] = []

    if not _is_not_stated(ref.get("value")):
        passed = _floats_close(agent.get("value"), ref["value"])
        checks.append((f"{prefix}.value", passed, str(ref["value"]), str(agent.get("value"))))

    ref_ci = ref.get("confidence_interval") or {}
    agent_ci = agent.get("confidence_interval") or {}
    for sub in ("lower", "upper"):
        if not _is_not_stated(ref_ci.get(sub)):
            passed = _floats_close(agent_ci.get(sub), ref_ci[sub])
            checks.append((
                f"{prefix}.ci.{sub}", passed,
                str(ref_ci[sub]), str(agent_ci.get(sub)),
            ))

    if not _is_not_stated(ref.get("direction")):
        agent_dir = str(agent.get("direction") or "").lower().strip()
        ref_dir = str(ref["direction"]).lower().strip()
        checks.append((f"{prefix}.direction", agent_dir == ref_dir, ref_dir, agent_dir))

    if ref.get("statistical_significance") is not None:
        ref_sig = _parse_bool(ref["statistical_significance"])
        agent_sig = _parse_bool(agent.get("statistical_significance"))
        passed = ref_sig is not None and agent_sig == ref_sig
        checks.append((
            f"{prefix}.significant", passed,
            str(ref_sig), str(agent_sig),
        ))

    return checks


def _score_numerical_result(agent: dict, ref: dict, prefix: str) -> Tuple[float, List[_Check]]:
    checks = _check_numerical_result(agent, ref, prefix)
    score = sum(c[1] for c in checks) / len(checks) if checks else 0.0
    return score, checks


def _score_against_gt(
    agent_data: dict, gt: dict
) -> Tuple[float, List[_Check]]:
    """Score agent output against one ground-truth variant.

    Returns (score, all_checks) where all_checks is the flat list of per-field checks.
    """
    ref_study = (gt.get("original_study") or {})
    agent_study = (agent_data.get("original_study") or {})

    all_checks: List[_Check] = []

    # sample_size
    ref_sample = (ref_study.get("data") or {}).get("sample_size")
    if not _is_not_stated(ref_sample):
        ref_n = _parse_int(ref_sample)
        agent_n = _parse_int((agent_study.get("data") or {}).get("sample_size"))
        if ref_n is not None:
            passed = ref_n == agent_n
            all_checks.append(("data.sample_size", passed, str(ref_n), str(agent_n)))

    # numerical_results
    ref_results = (ref_study.get("results") or {}).get("numerical_results") or []
    agent_results = (agent_study.get("results") or {}).get("numerical_results") or []

    for i, ref_r in enumerate(ref_results):
        prefix = f"numerical_results[{i}]"
        if not agent_results:
            # no agent results at all — mark every field as failed
            dummy_checks = _check_numerical_result({}, ref_r, prefix)
            all_checks.extend((f, False, exp, "missing") for f, _, exp, _ in dummy_checks)
            continue
        # pick best-matching agent result by score
        scored = [_score_numerical_result(a, ref_r, prefix) for a in agent_results]
        best_score, best_checks = max(scored, key=lambda x: x[0])
        all_checks.extend(best_checks)

    scores = [1.0 if c[1] else 0.0 for c in all_checks]
    score = sum(scores) / len(scores) if scores else 0.0
    return score, all_checks


def _format_feedback(score: float, checks: List[_Check]) -> str:
    lines = [f"Score: {score:.3f} ({sum(c[1] for c in checks)}/{len(checks)} fields correct)\n"]
    for field, passed, expected, got in checks:
        mark = "PASS" if passed else "FAIL"
        if passed:
            lines.append(f"  [{mark}] {field} = {got}")
        else:
            lines.append(f"  [{mark}] {field}: expected {expected!r}, got {got!r}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    gt_dir = log_dir / "groundtruth"
    example_id = example.get("id")

    output_file = workspace / "post_registration.json"
    if not output_file.exists():
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "feedback": "post_registration.json not found in workspace.",
        }

    try:
        agent_data = json.loads(output_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "feedback": f"post_registration.json is not valid JSON: {e}",
        }

    gt_file = gt_dir / "expected_post_registration.json"
    if not gt_file.exists():
        return {
            "workspace_dir": workspace_dir,
            "example_id": example_id,
            "score": 0.0,
            "feedback": "Ground truth not found — workspace setup may have failed.",
        }

    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    score, checks = _score_against_gt(agent_data, gt)

    gt2_file = gt_dir / "expected_post_registration_2.json"
    if gt2_file.exists():
        gt2 = json.loads(gt2_file.read_text(encoding="utf-8"))
        score2, checks2 = _score_against_gt(agent_data, gt2)
        if score2 > score:
            score, checks = score2, checks2

    return {
        "workspace_dir": workspace_dir,
        "example_id": example_id,
        "score": score,
        "feedback": _format_feedback(score, checks),
    }
