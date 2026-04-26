"""Workspace setup and MCP config for the woocommerce_stock_alert_s2l task.

Uses LOCA-bench's WooCommerce stock-alert preprocessing helpers to generate a
mock WooCommerce store, a Google Sheet, and an email account for the agent to
operate on.

Directory layout inside workspace_dir after setup:
    workspace_dir/
        admin_credentials.txt
        purchasing_manager_email.txt
        stock_alert_email_template.md
        local_db/
            woocommerce/
            emails/
            google_sheets/
    log_dir/
        groundtruth/task_artifacts/
            preprocess/woocommerce_products.json
            files/sheet_id.txt
"""

import os
import shutil
import sys
from pathlib import Path

_env_loca = os.environ.get("LOCA_BENCH_PATH")
LOCA_BENCH_PATH = Path(_env_loca) if _env_loca else Path(__file__).parent.parent.parent / "LOCA-bench"

TASK_INSTRUCTION = (
    "You need to read the inventory levels of WooCommerce products, check the "
    "current stock quantity (stock_quantity) for each product against the "
    "safety threshold (stock_threshold), identify all products with stock "
    "strictly below the threshold (stock_quantity < stock_threshold), and "
    "automatically update a Google Sheets purchase requisition list named "
    "WooCommerce Stock Alert (already in Google Sheets). For each low-stock "
    "product, record it in Google Sheets and send an individual email "
    "notification to the purchasing manager (the email address is in "
    "purchasing_manager_email.txt). You need to find all low-stock products, "
    "record them and send emails. The email template can be found in "
    "stock_alert_email_template.md. The email account credentials are in "
    "admin_credentials.txt in the workspace."
)


def _ensure_loca_path() -> None:
    loca_str = str(LOCA_BENCH_PATH)
    if loca_str not in sys.path:
        sys.path.insert(0, loca_str)


def setup_workspace(workspace_dir: str, log_dir: str, example: dict) -> None:
    _ensure_loca_path()
    from gem.envs.woocommerce_stock_alert_s2l.preprocess.main import (
        GoogleSheetsInitializer,
        WooCommerceProductSync,
        clear_all_email_folders,
        ensure_users_exist,
        generate_product_data,
        save_product_data,
    )
    from mcp_convert.mcps.email.database_utils import EmailDatabase
    from mcp_convert.mcps.woocommerce.init_database import initialize_database as init_woocommerce_db

    workspace = Path(workspace_dir)
    groundtruth_dir = Path(log_dir) / "groundtruth"
    task_artifacts_dir = groundtruth_dir / "task_artifacts"
    task_artifacts_dir.mkdir(parents=True, exist_ok=True)
    (task_artifacts_dir / "agent_workspace").mkdir(parents=True, exist_ok=True)
    (task_artifacts_dir / "preprocess").mkdir(parents=True, exist_ok=True)

    seed = example["seed"]
    num_low_stock = example["num_low_stock"]
    num_normal_stock = example["num_normal_stock"]

    woocommerce_db_dir = workspace / "local_db" / "woocommerce"
    email_db_dir = workspace / "local_db" / "emails"
    google_sheet_db_dir = workspace / "local_db" / "google_sheets"
    woocommerce_db_dir.mkdir(parents=True, exist_ok=True)
    email_db_dir.mkdir(parents=True, exist_ok=True)
    google_sheet_db_dir.mkdir(parents=True, exist_ok=True)

    products = generate_product_data(
        num_low_stock=num_low_stock,
        num_normal_stock=num_normal_stock,
        seed=seed,
    )
    products_file = task_artifacts_dir / "preprocess" / "woocommerce_products.json"
    save_product_data(products, products_file)

    init_woocommerce_db(str(woocommerce_db_dir), verbose=False, include_demo_data=False)

    email_db = EmailDatabase(data_dir=str(email_db_dir))
    ensure_users_exist(
        email_db,
        [
            {
                "email": "admin@woocommerce.local",
                "password": "admin123",
                "name": "WooCommerce Admin",
            }
        ],
    )
    clear_all_email_folders(email_db, "admin@woocommerce.local")

    wc_sync = WooCommerceProductSync(task_artifacts_dir, str(woocommerce_db_dir))
    wc_sync.sync_products()

    sheets_init = GoogleSheetsInitializer(task_artifacts_dir, str(google_sheet_db_dir))
    sheets_init.initialize_sheets()

    initial_workspace = (
        LOCA_BENCH_PATH / "gem" / "envs" / "woocommerce_stock_alert_s2l" / "initial_workspace"
    )
    for item in initial_workspace.iterdir():
        if item.is_file():
            shutil.copy2(item, workspace / item.name)


def setup_proposer_workspace(workspace_dir: str) -> None:
    workspace = Path(workspace_dir)
    (workspace / "local_db" / "woocommerce").mkdir(parents=True, exist_ok=True)
    (workspace / "local_db" / "emails").mkdir(parents=True, exist_ok=True)
    (workspace / "local_db" / "google_sheets").mkdir(parents=True, exist_ok=True)


def _build_server_config(server_script: Path, env: dict[str, str], docker_workspace: bool) -> dict:
    project_root = server_script.parent.parent.parent
    if docker_workspace or os.environ.get("LOCA_BENCH_PATH"):
        return {
            "command": "/workspace/.venv/bin/python",
            "args": [str(server_script)],
            "env": env,
        }
    return {
        "command": "uv",
        "args": ["--directory", str(project_root), "run", "python", str(server_script)],
        "env": env,
    }


def get_mcp_config(workspace_dir: str) -> dict:
    workspace = Path(workspace_dir).resolve()
    docker_workspace = str(workspace).startswith("/workspace/")
    loca_root = Path("/loca-bench") if docker_workspace else LOCA_BENCH_PATH
    mcp_root = loca_root / "mcp_convert" / "mcps"

    return {
        "mcpServers": {
            "woocommerce": _build_server_config(
                mcp_root / "woocommerce" / "server.py",
                {
                    "WOOCOMMERCE_DATA_DIR": str(workspace / "local_db" / "woocommerce"),
                    "LOCA_QUIET": "1",
                },
                docker_workspace,
            ),
            "google_sheet": _build_server_config(
                mcp_root / "google_sheet" / "server.py",
                {
                    "GOOGLE_SHEET_DATA_DIR": str(workspace / "local_db" / "google_sheets"),
                    "LOCA_QUIET": "1",
                },
                docker_workspace,
            ),
            "email": _build_server_config(
                mcp_root / "email" / "server.py",
                {
                    "EMAIL_DATA_DIR": str(workspace / "local_db" / "emails"),
                    "LOCA_QUIET": "1",
                },
                docker_workspace,
            ),
        }
    }
