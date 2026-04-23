"""Workspace setup and MCP config for the machine_operating_s2l task.

Uses LOCA-bench's machine_operating_s2l preprocessing to generate IoT sensor
data, upload it to a mock BigQuery dataset, and prepare the agent workspace.

Directory layout inside workspace_dir after setup:
    workspace_dir/
        machine_operating_parameters.xlsx  ← parameter file for the agent
        local_db/google_cloud/              ← mock BigQuery/Storage SQLite DB
    log_dir/
        groundtruth/anomaly_report.csv      ← ground truth (not visible to agent)
"""

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_env_loca = os.environ.get("LOCA_BENCH_PATH")
LOCA_BENCH_PATH = Path(_env_loca) if _env_loca else Path(__file__).parent.parent.parent / "LOCA-bench"


def _ensure_loca_path() -> None:
    loca_str = str(LOCA_BENCH_PATH)
    if loca_str not in sys.path:
        sys.path.insert(0, loca_str)


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    _ensure_loca_path()
    from mcp_convert.mcps.google_cloud.database_utils import GoogleCloudDatabase

    workspace = Path(workspace_dir).resolve()
    seed = example.get("seed", 42)
    hours = example.get("hours", 4)
    interval_minutes = example.get("interval_minutes", 5)
    anomaly_rate = example.get("anomaly_rate", 0.15)
    total_machines = example.get("total_machines", 10)
    total_sensors = str(example.get("total_sensors", "6"))

    gcloud_db_dir = workspace / "local_db" / "google_cloud"
    gcloud_db_dir.mkdir(parents=True, exist_ok=True)

    groundtruth_dir = (Path(log_dir) / "groundtruth").resolve()
    groundtruth_dir.mkdir(parents=True, exist_ok=True)

    env_dir = LOCA_BENCH_PATH / "gem" / "envs" / "machine_operating_s2l"
    construct_script = env_dir / "preprocess" / "construct_data.py"
    calc_script = env_dir / "preprocess" / "calculate_groundtruth.py"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        subprocess.run(
            [
                sys.executable, str(construct_script),
                "--hours", str(hours),
                "--interval", str(interval_minutes),
                "--anomaly-rate", str(anomaly_rate),
                "--seed", str(seed),
                "--output-dir", str(temp_path),
                "--total-machines", str(total_machines),
                "--total-sensors", total_sensors,
            ],
            check=True,
            capture_output=True,
            cwd=str(env_dir / "preprocess"),
        )

        sensor_csv = temp_path / "live_sensor_data.csv"
        params_xlsx = temp_path / "machine_operating_parameters.xlsx"

        gcloud_db = GoogleCloudDatabase(data_dir=str(gcloud_db_dir))
        project_id = "local-project"
        dataset_name = "machine_operating"

        gcloud_db.create_bigquery_dataset(project_id, dataset_name, {
            "location": "US",
            "description": "Machine operating dataset for IoT sensor data analysis",
            "labels": {},
        })

        rows = []
        with open(sensor_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                converted_row = {}
                for key, value in row.items():
                    try:
                        converted_row[key] = float(value) if "." in value else int(value)
                    except (ValueError, AttributeError):
                        converted_row[key] = value
                rows.append(converted_row)

        schema = []
        for key, value in rows[0].items():
            if isinstance(value, int):
                field_type = "INTEGER"
            elif isinstance(value, float):
                field_type = "FLOAT"
            else:
                field_type = "STRING"
            schema.append({"name": key, "type": field_type, "mode": "NULLABLE"})

        gcloud_db.create_bigquery_table(project_id, dataset_name, "live_sensor", {
            "schema": schema,
            "description": "Machine operating sensor data",
        })
        gcloud_db.insert_table_rows(project_id, dataset_name, "live_sensor", rows)

        shutil.copy2(str(params_xlsx), str(workspace / "machine_operating_parameters.xlsx"))

        groundtruth_file = groundtruth_dir / "anomaly_report.csv"
        subprocess.run(
            [
                sys.executable, str(calc_script),
                "--sensor-data", str(sensor_csv),
                "--parameters", str(params_xlsx),
                "--output", str(groundtruth_file),
            ],
            check=True,
            capture_output=True,
            cwd=str(env_dir / "preprocess"),
        )


def setup_proposer_workspace(workspace_dir: str) -> None:
    """Create an empty local_db so the MCP server can start in the proposer workspace."""
    (Path(workspace_dir) / "local_db" / "google_cloud").mkdir(parents=True, exist_ok=True)


def get_mcp_config(workspace_dir: str) -> dict:
    workspace = Path(workspace_dir).resolve()
    gcloud_db_dir = workspace / "local_db" / "google_cloud"
    docker_workspace = str(workspace).startswith("/workspace/")
    loca_root = Path("/loca-bench") if docker_workspace else LOCA_BENCH_PATH
    server_script = loca_root / "mcp_convert" / "mcps" / "google_cloud" / "server.py"
    project_root = loca_root / "mcp_convert"

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
