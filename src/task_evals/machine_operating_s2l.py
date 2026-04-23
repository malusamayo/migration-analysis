"""Evaluation for the machine_operating_s2l task.

Checks:
1. The agent created a Cloud Storage bucket named "iot_anomaly_reports".
2. The bucket contains a file matching anomaly_report*.csv.
3. The file's contents match the groundtruth anomaly report (precision and
   recall both ≥ 95%, with time tolerance of 60 s and reading tolerance 0.01).
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import dspy
import pandas as pd

LOCA_BENCH_PATH = Path(__file__).parent.parent.parent / "LOCA-bench"


def _ensure_loca_path() -> None:
    loca_str = str(LOCA_BENCH_PATH)
    if loca_str not in sys.path:
        sys.path.insert(0, loca_str)


def _normalize_timestamp(ts_str: str) -> datetime:
    cleaned = str(ts_str).strip()
    for suffix in ("+00:00", " UTC"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    if "+" in cleaned:
        cleaned = cleaned.split("+")[0]
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1]
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse timestamp: {ts_str!r}")


def _matches(agent_row: pd.Series, gt_row: pd.Series,
             time_tol: int = 60, reading_tol: float = 0.01) -> bool:
    try:
        t_diff = abs(
            (_normalize_timestamp(str(agent_row["timestamp"])) -
             _normalize_timestamp(str(gt_row["timestamp"]))).total_seconds()
        )
        return (
            t_diff <= time_tol
            and str(agent_row["machine_id"]).strip() == str(gt_row["machine_id"]).strip()
            and str(agent_row["sensor_type"]).strip() == str(gt_row["sensor_type"]).strip()
            and abs(float(agent_row["reading"]) - float(gt_row["reading"])) <= reading_tol
        )
    except Exception:
        return False


def _compute_precision_recall(agent_df: pd.DataFrame, gt_df: pd.DataFrame,
                               time_tol: int = 60, reading_tol: float = 0.01):
    matched_agent = sum(
        any(_matches(a_row, g_row, time_tol, reading_tol) for _, g_row in gt_df.iterrows())
        for _, a_row in agent_df.iterrows()
    )
    matched_gt = sum(
        any(_matches(g_row, a_row, time_tol, reading_tol) for _, a_row in agent_df.iterrows())
        for _, g_row in gt_df.iterrows()
    )
    precision = matched_agent / len(agent_df) if len(agent_df) > 0 else 0.0
    recall = matched_gt / len(gt_df) if len(gt_df) > 0 else 0.0
    return precision, recall


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    groundtruth_file = log_dir / "groundtruth" / "anomaly_report.csv"

    if not groundtruth_file.exists():
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "Ground truth not found — workspace setup may have failed.",
        }

    gcloud_db_dir = workspace / "local_db" / "google_cloud"
    if not gcloud_db_dir.exists():
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "Mock Google Cloud database not found.",
        }

    _ensure_loca_path()
    from mcp_convert.mcps.google_cloud.database_utils import GoogleCloudDatabase

    gcloud_db = GoogleCloudDatabase(data_dir=str(gcloud_db_dir))

    bucket = gcloud_db.get_storage_bucket("iot_anomaly_reports")
    if not bucket:
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "Bucket 'iot_anomaly_reports' not found.",
        }

    objects = gcloud_db.list_storage_objects("iot_anomaly_reports")
    report_files = [
        obj["name"]
        for obj in (objects or [])
        if obj.get("name", "").startswith("anomaly_report") and obj["name"].endswith(".csv")
    ]
    if not report_files:
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "No anomaly_report*.csv found in 'iot_anomaly_reports' bucket.",
        }

    selected = sorted(report_files)[-1]
    obj = gcloud_db.get_storage_object("iot_anomaly_reports", selected)
    content = (obj or {}).get("content", "")
    if not content:
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "anomaly_report.csv uploaded to bucket is empty.",
        }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        agent_tmp = f.name

    try:
        required_cols = {"timestamp", "machine_id", "sensor_type", "reading"}

        try:
            agent_df = pd.read_csv(agent_tmp)
        except Exception as e:
            return {"workspace_dir": workspace_dir, "score": 0.0, "feedback": f"Could not parse agent report: {e}"}

        missing = required_cols - set(agent_df.columns)
        if missing:
            return {
                "workspace_dir": workspace_dir,
                "score": 0.0,
                "feedback": f"Agent report missing columns: {sorted(missing)}",
            }

        gt_df = pd.read_csv(str(groundtruth_file))

        precision, recall = _compute_precision_recall(agent_df, gt_df)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if precision >= 0.95 and recall >= 0.95:
            return {
                "workspace_dir": workspace_dir,
                "score": 1.0,
                "feedback": (
                    f"Correct: precision={precision:.1%}, recall={recall:.1%}, F1={f1:.1%}."
                ),
            }
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": (
                f"Validation failed: precision={precision:.1%} "
                f"(agent={len(agent_df)}, matched={int(precision*len(agent_df))}), "
                f"recall={recall:.1%} "
                f"(gt={len(gt_df)}, matched={int(recall*len(gt_df))})."
            ),
        }
    finally:
        os.unlink(agent_tmp)
