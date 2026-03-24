"""Evaluation for the ab_testing task.

Checks:
1. record.csv exists and contains correct per-scenario conversion rates
   (within 0.05% tolerance vs. ground truth in log_dir/groundtruth/).
2. The overall conversion rate row is present.
3. The correct downstream action was taken in the mock Google Cloud backend:
   - B wins  → bucket "promo-assets-for-b" exists, no log entry required
   - A wins / tie → no bucket, log entry in "abtesting_logging"
"""

import ast
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import dspy

LOCA_BENCH_PATH = Path(__file__).parent.parent.parent / "LOCA-bench"


def _ensure_loca_path() -> None:
    loca_str = str(LOCA_BENCH_PATH)
    if loca_str not in sys.path:
        sys.path.insert(0, loca_str)


def _read_record_csv(path: Path) -> dict:
    records = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = (row.get("scenario") or "").strip()
            a_str = (row.get("A_conversion %") or "").strip()
            b_str = (row.get("B_conversion %") or "").strip()
            if not scenario or not a_str or not b_str:
                continue
            records[scenario] = {
                "A": float(a_str.replace("%", "")),
                "B": float(b_str.replace("%", "")),
            }
    return records


def _normalize(name: str) -> str:
    return name[3:] if name.startswith("ab_") else name


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    groundtruth_dir = log_dir / "groundtruth"

    record_file = workspace / "record.csv"
    if not record_file.exists():
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "record.csv not found in workspace.",
        }

    expected_file = groundtruth_dir / "expected_ratio.csv"
    if not expected_file.exists():
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "Ground truth not found — workspace setup may have failed.",
        }

    actual = _read_record_csv(record_file)
    expected = _read_record_csv(expected_file)

    if not actual:
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "record.csv is empty or malformed.",
        }

    norm_expected = {_normalize(k): v for k, v in expected.items()}
    norm_actual = {_normalize(k): v for k, v in actual.items()}

    errors = []
    tolerance = 0.05

    for scenario, exp_vals in norm_expected.items():
        if scenario not in norm_actual:
            errors.append(f"Missing scenario: {scenario}")
            continue
        act_vals = norm_actual[scenario]
        for version in ("A", "B"):
            diff = abs(act_vals[version] - exp_vals[version])
            if diff > tolerance:
                errors.append(
                    f"{scenario} version {version}: expected {exp_vals[version]:.3f}%"
                    f", got {act_vals[version]:.3f}% (diff {diff:.3f}%)"
                )

    extra = set(norm_actual) - set(norm_expected)
    if extra:
        errors.append(f"Unexpected scenarios: {sorted(extra)}")

    if errors:
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "Conversion rate validation failed:\n" + "\n".join(errors),
        }

    overall_key = next(
        (k for k in norm_actual if "overall" in k.lower()), None
    )
    if overall_key is None:
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "Overall row not found in record.csv.",
        }

    overall_A = norm_actual[overall_key]["A"]
    overall_B = norm_actual[overall_key]["B"]
    b_wins = overall_B > overall_A

    _ensure_loca_path()
    from mcp_convert.mcps.google_cloud.database_utils import GoogleCloudDatabase

    gcloud_db_dir = workspace / "local_db" / "google_cloud"
    if not gcloud_db_dir.exists():
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "Mock Google Cloud database not found.",
        }

    gcloud_db = GoogleCloudDatabase(data_dir=str(gcloud_db_dir))
    bucket_exists = gcloud_db.get_storage_bucket("promo-assets-for-b") is not None

    if b_wins:
        if not bucket_exists:
            return {
                "workspace_dir": workspace_dir,
                "score": 0.0,
                "feedback": "B won but bucket 'promo-assets-for-b' was not created.",
            }
        return {
            "workspace_dir": workspace_dir,
            "score": 1.0,
            "feedback": "Correct: B won, bucket created.",
        }

    # A wins or tie
    if bucket_exists:
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "A won/tie but bucket 'promo-assets-for-b' was incorrectly created.",
        }

    expected_log = {"status": "AB_Test_Concluded", "winner": "A", "action": "No_Change"}
    try:
        log_entries = gcloud_db.list_log_entries(
            filter_string='logName="abtesting_logging"', max_results=100
        )
    except Exception:
        log_entries = []

    found_log = False
    for entry in (log_entries or []):
        payload = entry.get("json_payload") or entry.get("text_payload")
        if isinstance(payload, dict):
            log_data = payload
        elif isinstance(payload, str):
            try:
                log_data = json.loads(payload)
            except Exception:
                try:
                    log_data = ast.literal_eval(payload)
                except Exception:
                    continue
        else:
            continue
        if log_data == expected_log:
            found_log = True
            break

    winner_str = "A wins" if overall_A > overall_B else "tie"
    if not found_log:
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": f"{winner_str}: expected log entry not found in abtesting_logging.",
        }

    return {
        "workspace_dir": workspace_dir,
        "score": 1.0,
        "feedback": f"Correct: {winner_str}, log entry written.",
    }
