"""Evaluation for the woocommerce_stock_alert_s2l task.

Checks:
1. The Google Sheet contains all and only low-stock products.
2. The agent sent one stock-alert email per low-stock product to the
   purchasing manager, following the benchmark template constraints.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import dspy

LOCA_BENCH_PATH = Path(__file__).parent.parent.parent / "LOCA-bench"


def _ensure_loca_path() -> None:
    loca_str = str(LOCA_BENCH_PATH)
    if loca_str not in sys.path:
        sys.path.insert(0, loca_str)


def run_single_instance_eval(
    workspace_dir: str,
    example: dict,
    lm: Optional[dspy.LM] = None,
) -> Dict[str, Any]:
    workspace = Path(workspace_dir)
    log_dir = workspace.parent / f"{workspace.name}_logs"
    task_artifacts_dir = log_dir / "groundtruth" / "task_artifacts"

    products_file = task_artifacts_dir / "preprocess" / "woocommerce_products.json"
    sheet_id_file = task_artifacts_dir / "files" / "sheet_id.txt"
    if not products_file.exists() or not sheet_id_file.exists():
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "Ground truth not found — workspace setup may have failed.",
        }

    woocommerce_db_dir = workspace / "local_db" / "woocommerce"
    email_db_dir = workspace / "local_db" / "emails"
    google_sheet_db_dir = workspace / "local_db" / "google_sheets"
    if not woocommerce_db_dir.exists() or not email_db_dir.exists() or not google_sheet_db_dir.exists():
        return {
            "workspace_dir": workspace_dir,
            "score": 0.0,
            "feedback": "Mock task databases not found in workspace/local_db.",
        }

    _ensure_loca_path()
    from gem.envs.woocommerce_stock_alert_s2l.evaluation.evaluate_updated_stock_alert import StockAlertEvaluator
    from mcp_convert.mcps.email.database_utils import EmailDatabase
    from mcp_convert.mcps.google_sheet.database_utils import GoogleSheetDatabase
    from mcp_convert.mcps.woocommerce.database_utils import WooCommerceDatabase

    woocommerce_db = WooCommerceDatabase(data_dir=str(woocommerce_db_dir))
    email_db = EmailDatabase(data_dir=str(email_db_dir))
    google_sheet_db = GoogleSheetDatabase(data_dir=str(google_sheet_db_dir))

    evaluator = StockAlertEvaluator(
        agent_workspace=str(task_artifacts_dir / "agent_workspace"),
        email_db=email_db,
        google_sheet_db=google_sheet_db,
        woocommerce_db=woocommerce_db,
    )
    results = evaluator.run_evaluation()
    overall = results.get("overall", {})
    if overall.get("passed", False):
        return {
            "workspace_dir": workspace_dir,
            "score": 1.0,
            "feedback": "Correct: Google Sheet and stock-alert emails match the benchmark checks.",
        }

    failures = []
    for key in ("google_sheets_update", "email_notifications"):
        component = results.get(key, {})
        if not component.get("passed", False):
            failures.append(component.get("message", key))
    return {
        "workspace_dir": workspace_dir,
        "score": 0.0,
        "feedback": "Validation failed: " + " | ".join(failures),
    }
