"""Workspace setup and MCP config for the ab_testing task.

Uses LOCA-bench's ABTestingDataGenerator and the mock Google Cloud SQLite
backend (mcp_convert) — no external services or new dependencies required.

Directory layout inside workspace_dir after setup:
    workspace_dir/
        record.csv            ← blank template for the agent to fill in
        local_db/google_cloud/  ← mock BigQuery/Storage/Logging SQLite DB
    log_dir/
        groundtruth/expected_ratio.csv  ← ground truth (not visible to agent)
"""

import os
import shutil
import sys
from pathlib import Path

_env_loca = os.environ.get("LOCA_BENCH_PATH")
LOCA_BENCH_PATH = Path(_env_loca) if _env_loca else Path(__file__).parent.parent.parent / "LOCA-bench"

TASK_INSTRUCTION = (
    "The A/B test for our new homepage has concluded, and the raw clickstream data "
    "has been stored in the `ab_testing` dataset in BigQuery. Analyze this data to "
    "calculate the conversion rate for each scenario as well as the overall conversion "
    "rate, which should be labeled `overall (total_store_views/total_clicks)`. Record "
    "these results in `record.csv`, following the same format used in that file — do "
    "not change column names. After completing the analysis, determine which version "
    "('A' or 'B') has the highest overall conversion rate, i.e., the overall conversion "
    "rate is defined as the arithmetic mean of the per-scenario conversion rates. If "
    "version B outperforms, immediately create a new Cloud Storage bucket named "
    "`promo-assets-for-b` for the full promotion, and you do not need to write any log "
    "entry in this process. If version A wins or the results are a tie, no bucket "
    "creation is required, but a log entry with the message `{'status': "
    "'AB_Test_Concluded', 'winner': 'A', 'action': 'No_Change'}` must be written to "
    "the `abtesting_logging` bucket."
)


def _ensure_loca_path() -> None:
    loca_str = str(LOCA_BENCH_PATH)
    if loca_str not in sys.path:
        sys.path.insert(0, loca_str)


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    _ensure_loca_path()
    from gem.envs.ab_testing_s2l.generate_ab_data import ABTestingDataGenerator
    from mcp_convert.mcps.google_cloud.database_utils import GoogleCloudDatabase

    workspace = Path(workspace_dir)
    seed = example["seed"]
    num_scenarios = example["num_scenarios"]
    num_days = example["num_days"]
    difficulty = example["difficulty"]

    gcloud_db_dir = workspace / "local_db" / "google_cloud"
    gcloud_db_dir.mkdir(parents=True, exist_ok=True)

    groundtruth_dir = Path(log_dir) / "groundtruth"
    groundtruth_dir.mkdir(parents=True, exist_ok=True)

    generator = ABTestingDataGenerator(seed=seed)
    result = generator.generate_scenarios(
        num_scenarios=num_scenarios,
        num_days=num_days,
        difficulty=difficulty,
    )
    scenarios = result["scenarios"]

    generator.save_expected_ratio(scenarios, groundtruth_dir / "expected_ratio.csv")

    gcloud_db = GoogleCloudDatabase(data_dir=str(gcloud_db_dir))
    project_id = "local-project"
    dataset_id = "ab_testing"

    gcloud_db.create_bigquery_dataset(project_id, dataset_id, {
        "location": "US",
        "description": "A/B testing dataset for conversion rate analysis",
        "labels": {},
    })

    schema = [
        {"name": "time_window",   "type": "STRING",  "mode": "NULLABLE"},
        {"name": "A_clicks",      "type": "INTEGER", "mode": "NULLABLE"},
        {"name": "A_store_views", "type": "INTEGER", "mode": "NULLABLE"},
        {"name": "B_clicks",      "type": "INTEGER", "mode": "NULLABLE"},
        {"name": "B_store_views", "type": "INTEGER", "mode": "NULLABLE"},
    ]
    for scenario in scenarios:
        table_name = f"ab_{scenario['name']}"
        gcloud_db.create_bigquery_table(project_id, dataset_id, table_name, {
            "schema": schema,
            "description": f"Clickstream data for scenario {scenario['name']}",
        })
        gcloud_db.insert_table_rows(
            project_id, dataset_id, table_name, scenario["data_rows"]
        )

    initial_workspace = (
        LOCA_BENCH_PATH / "gem" / "envs" / "ab_testing_s2l" / "initial_workspace"
    )
    for item in initial_workspace.iterdir():
        if item.is_file():
            shutil.copy2(item, workspace / item.name)


def get_mcp_config(workspace_dir: str) -> dict:
    workspace = Path(workspace_dir).resolve()
    gcloud_db_dir = workspace / "local_db" / "google_cloud"
    docker_workspace = str(workspace).startswith("/workspace/")
    loca_root = Path("/loca-bench") if docker_workspace else LOCA_BENCH_PATH
    server_script = loca_root / "mcp_convert" / "mcps" / "google_cloud" / "server.py"
    project_root = loca_root / "mcp_convert"

    # Docker-backed runs serialize MCP config on the host, so detect container
    # workspaces by path rather than relying on runtime env from the agent server.
    # The subprocess env dict replaces the parent env, so use an absolute python
    # path inside the image instead of depending on `uv` being on PATH there.
    if docker_workspace or os.environ.get("LOCA_BENCH_PATH"):
        command = "/workspace/.venv/bin/python"
        args = [str(server_script)]
    else:
        command = "uv"
        args = ["--directory", str(project_root), "run", "python", str(server_script)]

    return {
        "mcpServers": {
            "google_cloud": {
                "command": command,
                "args": args,
                "env": {
                    "GOOGLE_CLOUD_DATA_DIR": str(gcloud_db_dir),
                    "LOCA_QUIET": "1",
                },
            }
        }
    }
