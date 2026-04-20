#!/usr/bin/env python3

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REQUIRED_COLUMNS = (
    "task_id",
    "model_name",
    "max_examples",
    "max_cost",
    "use_adaptation_guide",
    "reflection_lm",
)

DEFAULT_MANIFEST_COLUMNS = (
    "task_id",
    "model_name",
    "max_examples",
    "max_cost",
    "use_adaptation_guide",
    "reflection_lm",
    "prompt_name",
    "seed",
    "num_exploration",
)

HEADER_ALIASES = {
    "task": "task_id",
    "task_id": "task_id",
    "task_name": "task_id",
    "task_lm": "model_name",
    "model": "model_name",
    "model_name": "model_name",
    "lm": "model_name",
    "n": "max_examples",
    "num_examples": "max_examples",
    "max_examples": "max_examples",
    "budget": "max_cost",
    "budget_$": "max_cost",
    "budget_dollar": "max_cost",
    "max_cost": "max_cost",
    "cost_budget": "max_cost",
    "use_adaptation": "use_adaptation_guide",
    "use_adaptation_guide": "use_adaptation_guide",
    "adaptation": "use_adaptation_guide",
    "reflection_lm": "reflection_lm",
    "reflection_model": "reflection_lm",
    "reflection": "reflection_lm",
    "prompt": "prompt_name",
    "prompt_name": "prompt_name",
    "seed": "seed",
    "num_strategies": "num_exploration",
    "strategies": "num_exploration",
    "num_exploration": "num_exploration",
}

POST_GEPA_COLUMNS = [
    "Task",
    "Run",
    "task_lm",
    "prompt_name",
    "seed",
    "use_adaptation",
    "reflection_lm",
    "train_examples",
    "val_examples",
    "gepa_best_iteration",
    "gepa_val_score",
    "rerun_rollout_version",
    "rerun_cost",
    "rerun_score",
    "rerun_train_score",
    "rerun_val_score",
    "rerun_train_rollout_0",
    "rerun_train_rollout_1",
    "rerun_train_rollout_2",
    "rerun_val_rollout_0",
    "rerun_val_rollout_1",
    "rerun_val_rollout_2",
]

RESULT_TABLE_COLUMNS = [
    "Task",
    "Run",
    "task_lm",
    "N",
    "budget",
    "use_adaptation",
    "reflection_lm",
    "initial_val_score",
    "best_val_score",
    "iter",
    "cost",
    "time_mins",
]

STRICT_USED_CONFIG_KEYS = (
    "task_id",
    "model_name",
    "prompt_name",
    "max_examples",
    "train_ratio",
    "eval_lm",
    "reflection_lm",
    "seed",
    "data_path",
    "agent_batch_size",
    "eval_batch_size",
    "use_docker",
    "server_image",
    "max_time",
)

RESULT_HEADER_RE = re.compile(r"Task: (?P<task>\S+), Model: (?P<model>\S+), Prompt: (?P<prompt>\S+)")
RESULT_DATASET_RE = re.compile(r"Dataset: (?P<count>\d+) total")
RESULT_INITIAL_SCORE_RE = re.compile(
    r"Iteration 0: Base program full valset score: (?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
RESULT_FIRST_BEST_PATTERNS = [
    re.compile(
        r"Iteration 0: Base program full valset score: (?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ),
    re.compile(
        r"Iteration (?P<iter>\d+): Found a better program on the valset with score (?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ),
    re.compile(
        r"Iteration (?P<iter>\d+): Valset score for new program: (?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ),
    re.compile(
        r"Iteration (?P<iter>\d+): Best score on valset: (?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    ),
]
RESULT_COST_RE = re.compile(r"\[cost\] total=\$(?P<cost>\d+(?:\.\d+)?)")
TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ")


@dataclass
class PreparedRun:
    index: int
    task_id: str
    model_name: str
    prompt_name: str
    max_examples: int
    max_cost: float
    use_adaptation_guide: bool
    reflection_lm: str
    num_exploration: int
    seed: int
    run_dir: str
    optimize_config_path: str
    optimize_log_path: str
    post_run_config_path: str
    post_collect_log_path: str
    post_evaluate_log_path: str
    post_summary_path: str
    base_gepa_config_path: str
    base_run_config_path: str


@dataclass
class BestHit:
    iteration: int
    cost: float
    time_minutes: int


def normalize_header(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return HEADER_ALIASES.get(normalized, normalized)


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def split_table_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []
    if "\t" in stripped:
        return [cell.strip() for cell in re.split(r"\t+", stripped) if cell.strip()]
    if stripped.count(",") >= 2:
        return [cell.strip() for cell in next(csv.reader([stripped]))]
    return [cell.strip() for cell in re.split(r"\s{2,}", stripped) if cell.strip()]


def load_manifest_text(path: str | None) -> str:
    if path is None or path == "-":
        print("Reading manifest from stdin...")
        print("(End input with Ctrl+D on Unix or Ctrl+Z on Windows)")
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


def clean_manifest_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("```"):
            continue
        lines.append(raw_line.rstrip())
    return lines


def is_int_string(value: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+", value.strip()))


def infer_headerless_columns(first_line_values: list[str]) -> list[str]:
    if len(first_line_values) < len(REQUIRED_COLUMNS):
        raise ValueError(
            f"Manifest line 1 has {len(first_line_values)} columns, expected at least {len(REQUIRED_COLUMNS)}"
        )
    if len(first_line_values) > len(DEFAULT_MANIFEST_COLUMNS):
        raise ValueError(
            f"Manifest line 1 has {len(first_line_values)} columns, expected at most "
            f"{len(DEFAULT_MANIFEST_COLUMNS)} for headerless input"
        )

    extra_values = first_line_values[len(REQUIRED_COLUMNS) :]
    header = list(REQUIRED_COLUMNS)
    if not extra_values:
        return header
    if len(extra_values) == 1:
        header.append("num_exploration" if is_int_string(extra_values[0]) else "prompt_name")
        return header
    if len(extra_values) == 2:
        if is_int_string(extra_values[0]) and is_int_string(extra_values[1]):
            header.extend(["seed", "num_exploration"])
        elif is_int_string(extra_values[1]):
            header.extend(["prompt_name", "num_exploration"])
        else:
            header.extend(["prompt_name", "seed"])
        return header
    header.extend(["prompt_name", "seed", "num_exploration"])
    return header


def infer_manifest_header(first_line_values: list[str]) -> tuple[list[str], int]:
    normalized_values = [normalize_header(cell) for cell in first_line_values]
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized_values]
    if not missing:
        return normalized_values, 1

    return infer_headerless_columns(first_line_values), 0


def parse_manifest_table(text: str) -> list[dict[str, Any]]:
    lines = clean_manifest_lines(text)
    if not lines:
        raise ValueError("Manifest is empty")

    first_line_values = split_table_line(lines[0])
    header, data_start_index = infer_manifest_header(first_line_values)
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            f"{', '.join(missing)}. Headerless input must follow column order: "
            f"{', '.join(DEFAULT_MANIFEST_COLUMNS)}"
        )

    records = []
    for line_number, line in enumerate(lines[data_start_index:], start=data_start_index + 1):
        values = first_line_values if line_number == 1 else split_table_line(line)
        if not values:
            continue
        if len(values) != len(header):
            raise ValueError(
                f"Manifest line {line_number} has {len(values)} columns, expected {len(header)}"
            )
        raw_record = dict(zip(header, values, strict=True))
        record = {
            "task_id": raw_record["task_id"].strip(),
            "model_name": raw_record["model_name"].strip(),
            "max_examples": int(raw_record["max_examples"]),
            "max_cost": float(raw_record["max_cost"]),
            "use_adaptation_guide": parse_bool(raw_record["use_adaptation_guide"]),
            "reflection_lm": raw_record["reflection_lm"].strip(),
        }
        if "num_exploration" in raw_record and raw_record["num_exploration"].strip():
            record["num_exploration"] = int(raw_record["num_exploration"])
        if "prompt_name" in raw_record and raw_record["prompt_name"].strip():
            record["prompt_name"] = raw_record["prompt_name"].strip()
        if "seed" in raw_record and raw_record["seed"].strip():
            record["seed"] = int(raw_record["seed"])
        records.append(record)
    return records


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def strict_load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required YAML file: {path}")
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def strict_load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def format_budget(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def format_score(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group("timestamp"), "%Y-%m-%d %H:%M:%S")


def approximately_equal(lhs: float, rhs: float, tolerance: float = 1e-9) -> bool:
    return abs(lhs - rhs) <= tolerance


def get_base_gepa_config_path(task_id: str) -> Path:
    path = Path("tasks") / task_id / "gepa_optimize.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing GEPA base config: {path}")
    return path


def get_base_run_config_path(task_id: str) -> Path:
    path = Path("tasks") / task_id / "run.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing run config template: {path}")
    return path


def load_base_config(task_id: str) -> tuple[Path, dict[str, Any]]:
    config_path = get_base_gepa_config_path(task_id)
    return config_path, strict_load_yaml(config_path)


def family_key(task_id: str, model_name: str, prompt_name: str) -> tuple[str, str, str]:
    return task_id, model_name, prompt_name


def collect_existing_seeds(task_id: str, model_name: str, prompt_name: str) -> set[int]:
    existing = set()
    base_dir = Path("results") / task_id / f"{model_name}_{prompt_name}" / "gepa"
    if not base_dir.exists():
        return existing
    for child in base_dir.iterdir():
        match = re.fullmatch(r"seed(\d+)", child.name)
        if match:
            existing.add(int(match.group(1)))
    return existing


def build_batch_dir(requested_dir: str | None) -> Path:
    if requested_dir:
        batch_dir = Path(requested_dir)
        if batch_dir.exists():
            raise FileExistsError(f"Batch directory already exists: {batch_dir}")
        batch_dir.mkdir(parents=True, exist_ok=False)
        return batch_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = Path("generated") / "gepa_batches" / timestamp
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = Path("generated") / "gepa_batches" / f"{timestamp}_{suffix:02d}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def build_prepared_runs(
    manifest_text: str,
    records: list[dict[str, Any]],
    batch_dir: Path,
    max_parallel: int,
) -> list[PreparedRun]:
    configs_dir = batch_dir / "configs"
    logs_dir = batch_dir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)
    (batch_dir / "raw_manifest.txt").write_text(manifest_text, encoding="utf-8")

    used_seeds: dict[tuple[str, str, str], set[int]] = {}
    prepared_runs = []

    for index, record in enumerate(records):
        base_gepa_config_path, base_gepa_config = load_base_config(record["task_id"])
        base_run_config_path = get_base_run_config_path(record["task_id"])
        prompt_name = record.get("prompt_name") or base_gepa_config.get("prompt_name")
        if prompt_name is None:
            raise ValueError(f"Task {record['task_id']} base config does not define prompt_name")

        key = family_key(record["task_id"], record["model_name"], prompt_name)
        if key not in used_seeds:
            used_seeds[key] = collect_existing_seeds(*key)

        if "seed" in record:
            seed = record["seed"]
            if seed in used_seeds[key]:
                raise ValueError(
                    f"Seed {seed} already exists for {record['task_id']} / {record['model_name']} / {prompt_name}"
                )
        else:
            seed = 0
            while seed in used_seeds[key]:
                seed += 1
        used_seeds[key].add(seed)

        run_dir = (
            Path("results")
            / record["task_id"]
            / f"{record['model_name']}_{prompt_name}"
            / "gepa"
            / f"seed{seed}"
        )

        config = dict(base_gepa_config)
        config["task_id"] = record["task_id"]
        config["model_name"] = record["model_name"]
        config["prompt_name"] = prompt_name
        config["max_examples"] = record["max_examples"]
        config["max_cost"] = record["max_cost"]
        config["use_adaptation_guide"] = record["use_adaptation_guide"]
        config["reflection_lm"] = record["reflection_lm"]
        config["num_exploration"] = record.get("num_exploration", 1)
        config["seed"] = seed
        config["run_dir"] = str(run_dir)

        config_name = (
            f"{index:02d}_{slugify(record['task_id'])}_{slugify(record['model_name'])}"
            f"_{slugify(prompt_name)}_seed{seed}_adapt-{format_bool(record['use_adaptation_guide'])}.yaml"
        )
        optimize_config_path = configs_dir / config_name
        with open(optimize_config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)

        log_prefix = logs_dir / optimize_config_path.stem
        prepared_runs.append(
            PreparedRun(
                index=index,
                task_id=record["task_id"],
                model_name=record["model_name"],
                prompt_name=prompt_name,
                max_examples=record["max_examples"],
                max_cost=record["max_cost"],
                use_adaptation_guide=record["use_adaptation_guide"],
                reflection_lm=record["reflection_lm"],
                num_exploration=record.get("num_exploration", 1),
                seed=seed,
                run_dir=str(run_dir),
                optimize_config_path=str(optimize_config_path),
                optimize_log_path=str(log_prefix.with_name(f"{log_prefix.name}_optimize.log")),
                post_run_config_path=str(run_dir / "shared" / "config" / "post_gepa_run.yaml"),
                post_collect_log_path=str(log_prefix.with_name(f"{log_prefix.name}_collect.log")),
                post_evaluate_log_path=str(log_prefix.with_name(f"{log_prefix.name}_evaluate.log")),
                post_summary_path=str(run_dir / "shared" / "config" / "post_gepa_summary.json"),
                base_gepa_config_path=str(base_gepa_config_path),
                base_run_config_path=str(base_run_config_path),
            )
        )

    batch_metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "max_parallel": max_parallel,
        "runs": [asdict(run) for run in prepared_runs],
    }
    with open(batch_dir / "batch.json", "w", encoding="utf-8") as handle:
        json.dump(batch_metadata, handle, indent=2)

    return prepared_runs


def print_prepared_runs(runs: list[PreparedRun], batch_dir: Path, max_parallel: int) -> None:
    print(f"Prepared batch: {batch_dir}")
    print(f"Parallel launch limit: {max_parallel}")
    print()

    headers = ["idx", "task", "model", "prompt", "seed", "N", "budget", "adapt", "exploration", "reflection_lm"]
    rows = [
        [
            str(run.index),
            run.task_id,
            run.model_name,
            run.prompt_name,
            str(run.seed),
            str(run.max_examples),
            format_budget(run.max_cost),
            format_bool(run.use_adaptation_guide),
            str(run.num_exploration),
            run.reflection_lm,
        ]
        for run in runs
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))

    print()
    for run in runs:
        print(f"[{run.index}] optimize_config = {run.optimize_config_path}")
        print(f"    run_dir = {run.run_dir}")
        print(f"    optimize_log = {run.optimize_log_path}")
        print(f"    post_run_config = {run.post_run_config_path}")
        print(f"    post_summary = {run.post_summary_path}")


def read_batch(batch_dir: Path) -> tuple[dict[str, Any], list[PreparedRun]]:
    batch_path = batch_dir / "batch.json"
    if not batch_path.exists():
        raise FileNotFoundError(f"Missing batch.json in {batch_dir}")
    with open(batch_path, encoding="utf-8") as handle:
        data = json.load(handle)
    runs = [PreparedRun(**run_data) for run_data in data["runs"]]
    return data, runs


def prompt_for_confirmation(message: str) -> bool:
    response = input(f"{message} [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def log_launcher_event(
    launcher_log_path: Path,
    launcher_log_lock: threading.Lock,
    event: str,
) -> None:
    with launcher_log_lock:
        with open(launcher_log_path, "a", encoding="utf-8") as launcher_log:
            launcher_log.write(f"[{datetime.now().isoformat(timespec='seconds')}] {event}\n")


def run_logged_command(
    command: list[str],
    log_path: Path,
    prefix: str,
    console_lock: threading.Lock,
    launcher_log_path: Path,
    launcher_log_lock: threading.Lock,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_display = " ".join(command)
    log_launcher_event(launcher_log_path, launcher_log_lock, f"START {prefix} command={command_display}")

    with open(log_path, "w", encoding="utf-8") as run_log:
        run_log.write(f"$ {command_display}\n")
        run_log.flush()

        process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            run_log.write(line)
            run_log.flush()
            with console_lock:
                sys.stdout.write(f"[{prefix}] {line}")
                sys.stdout.flush()

        return_code = process.wait()

    status = "COMPLETED" if return_code == 0 else "FAILED"
    log_launcher_event(
        launcher_log_path,
        launcher_log_lock,
        f"{status} {prefix} return_code={return_code} log={log_path}",
    )
    return return_code


def build_train_val_data(used_config: dict[str, Any], output_path: Path) -> tuple[Path, int, int]:
    data_path = Path(used_config["data_path"])
    with open(data_path, encoding="utf-8") as handle:
        data = json.load(handle)

    max_examples = used_config["max_examples"]
    data = data[:max_examples]
    split_idx = max(1, int(len(data) * used_config["train_ratio"]))
    trainset = data[:split_idx]
    valset = data[split_idx:] if split_idx < len(data) else data
    rerun_evalset = trainset + valset

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(rerun_evalset, handle, indent=2)

    return output_path, len(trainset), len(valset)


def build_post_gepa_rollout_version(used_config: dict[str, Any]) -> str:
    return f"gepa_seed{used_config['seed']}_best"


def create_post_gepa_run_config(run: PreparedRun) -> tuple[dict[str, Any], Path]:
    run_dir = Path(run.run_dir)
    used_config_path = run_dir / "shared" / "config" / "used_config.yaml"
    summary_path = run_dir / "shared" / "config" / "optimization_summary.json"
    best_config_path = run_dir / "shared" / "config" / "best_config.py"
    base_run_config_path = Path(run.base_run_config_path)

    used_config = strict_load_yaml(used_config_path)
    for key in STRICT_USED_CONFIG_KEYS:
        if key not in used_config:
            raise ValueError(f"Missing required key '{key}' in {used_config_path}")
    summary = strict_load_json(summary_path)
    if "best_iteration" not in summary:
        raise ValueError(f"Missing best iteration metadata in {summary_path}")
    if not best_config_path.exists():
        raise FileNotFoundError(f"Missing best config: {best_config_path}")

    base_run_config = strict_load_yaml(base_run_config_path)
    evalset_data_path, train_examples, val_examples = build_train_val_data(
        used_config,
        run_dir / "shared" / "config" / "post_gepa_evalset.json",
    )

    rollout_version = build_post_gepa_rollout_version(used_config)
    post_run_config = dict(base_run_config)
    post_run_config["task_id"] = used_config["task_id"]
    post_run_config["model_name"] = used_config["model_name"]
    post_run_config["prompt_name"] = used_config["prompt_name"]
    post_run_config["max_examples"] = train_examples + val_examples
    post_run_config["n_responses"] = 3
    post_run_config["agent_batch_size"] = used_config["agent_batch_size"]
    post_run_config["eval_batch_size"] = used_config["eval_batch_size"]
    post_run_config["resume"] = False
    post_run_config["rollout_version"] = rollout_version
    post_run_config["data_path"] = str(evalset_data_path)
    post_run_config["agent_file"] = str(best_config_path)
    post_run_config["eval_lm"] = used_config["eval_lm"]
    post_run_config["use_docker"] = used_config["use_docker"]
    post_run_config["server_image"] = used_config["server_image"]
    post_run_config["max_time"] = used_config["max_time"]
    if "docker_network" in used_config:
        post_run_config["docker_network"] = used_config["docker_network"]

    post_run_config["source_gepa_run_dir"] = str(run_dir)
    post_run_config["source_gepa_config"] = str(used_config_path)
    post_run_config["source_train_examples"] = train_examples
    post_run_config["source_val_examples"] = val_examples

    post_run_config_path = Path(run.post_run_config_path)
    post_run_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(post_run_config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(post_run_config, handle, sort_keys=False)
    return post_run_config, post_run_config_path


def summarize_eval_scores(scores: list[float], n_rollouts: int) -> dict[str, Any]:
    if len(scores) % n_rollouts != 0:
        raise ValueError(
            f"Expected score count to be divisible by n_rollouts={n_rollouts}, got {len(scores)}"
        )

    num_examples = len(scores) // n_rollouts
    scores_by_rollout = []
    for rollout_id in range(n_rollouts):
        rollout_scores = [scores[i * n_rollouts + rollout_id] for i in range(num_examples)]
        scores_by_rollout.append(sum(rollout_scores) / len(rollout_scores))

    return {
        "avg_score": sum(scores) / len(scores),
        "scores": scores,
        "num_results": len(scores),
        "scores_by_rollout": scores_by_rollout,
    }


def summarize_run_costs(costs: list[float], n_rollouts: int) -> dict[str, Any]:
    if len(costs) % n_rollouts != 0:
        raise ValueError(
            f"Expected cost count to be divisible by n_rollouts={n_rollouts}, got {len(costs)}"
        )

    num_examples = len(costs) // n_rollouts
    total_costs_by_rollout = []
    avg_costs_by_rollout = []
    for rollout_id in range(n_rollouts):
        rollout_costs = [costs[i * n_rollouts + rollout_id] for i in range(num_examples)]
        total_costs_by_rollout.append(sum(rollout_costs))
        avg_costs_by_rollout.append(sum(rollout_costs) / len(rollout_costs))

    return {
        "total_cost": sum(costs),
        "avg_cost": sum(costs) / len(costs),
        "costs": costs,
        "num_results": len(costs),
        "total_costs_by_rollout": total_costs_by_rollout,
        "avg_costs_by_rollout": avg_costs_by_rollout,
    }


def load_eval_summary(
    eval_results_path: Path,
    train_examples: int,
    val_examples: int,
    n_rollouts: int,
) -> dict[str, Any]:
    if not eval_results_path.exists():
        raise FileNotFoundError(f"Missing evaluation results: {eval_results_path}")
    with open(eval_results_path, encoding="utf-8") as handle:
        results = yaml.safe_load(handle)
    if not isinstance(results, list) or not results:
        raise ValueError(f"Expected non-empty eval result list in {eval_results_path}")

    def _workspace_sort_key(r):
        m = re.search(r"example(\d+)_rollout(\d+)", r["workspace_dir"])
        return (int(m.group(1)), int(m.group(2)))

    results = sorted(results, key=_workspace_sort_key)
    scores = [float(result["score"]) for result in results]
    expected_num_results = (train_examples + val_examples) * n_rollouts
    if len(scores) != expected_num_results:
        raise ValueError(
            f"Expected {expected_num_results} eval results in {eval_results_path}, got {len(scores)}"
        )

    train_result_count = train_examples * n_rollouts
    train_scores = scores[:train_result_count]
    val_scores = scores[train_result_count:]
    return {
        "overall": summarize_eval_scores(scores, n_rollouts),
        "train": summarize_eval_scores(train_scores, n_rollouts),
        "val": summarize_eval_scores(val_scores, n_rollouts),
    }


def load_run_cost_summary(
    run_results_path: Path,
    train_examples: int,
    val_examples: int,
    n_rollouts: int,
) -> dict[str, Any]:
    if not run_results_path.exists():
        raise FileNotFoundError(f"Missing run results: {run_results_path}")
    with open(run_results_path, encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list) or not results:
        raise ValueError(f"Expected non-empty run result list in {run_results_path}")

    costs = [float(result["run_result"]["metrics"]["accumulated_cost"]) for result in results]
    expected_num_results = (train_examples + val_examples) * n_rollouts
    if len(costs) != expected_num_results:
        raise ValueError(
            f"Expected {expected_num_results} run results in {run_results_path}, got {len(costs)}"
        )

    train_result_count = train_examples * n_rollouts
    train_costs = costs[:train_result_count]
    val_costs = costs[train_result_count:]
    return {
        "overall": summarize_run_costs(costs, n_rollouts),
        "train": summarize_run_costs(train_costs, n_rollouts),
        "val": summarize_run_costs(val_costs, n_rollouts),
    }


def build_post_gepa_summary(run: PreparedRun) -> dict[str, Any]:
    run_dir = Path(run.run_dir)
    used_config = strict_load_yaml(run_dir / "shared" / "config" / "used_config.yaml")
    optimization_summary = strict_load_json(run_dir / "shared" / "config" / "optimization_summary.json")
    post_run_config = strict_load_yaml(Path(run.post_run_config_path))
    eval_results_path = (
        Path("results")
        / used_config["task_id"]
        / f"{used_config['model_name']}_{used_config['prompt_name']}"
        / "rollouts"
        / post_run_config["rollout_version"]
        / "eval_results.yaml"
    )
    run_results_path = (
        Path("results")
        / used_config["task_id"]
        / f"{used_config['model_name']}_{used_config['prompt_name']}"
        / "rollouts"
        / post_run_config["rollout_version"]
        / "run.json"
    )
    train_examples = int(post_run_config["source_train_examples"])
    val_examples = int(post_run_config["source_val_examples"])
    n_rollouts = int(post_run_config["n_responses"])
    eval_summary = load_eval_summary(eval_results_path, train_examples, val_examples, n_rollouts)
    run_cost_summary = load_run_cost_summary(
        run_results_path,
        train_examples,
        val_examples,
        n_rollouts,
    )

    return {
        "task_id": used_config["task_id"],
        "model_name": used_config["model_name"],
        "prompt_name": used_config["prompt_name"],
        "seed": used_config["seed"],
        "use_adaptation_guide": used_config["use_adaptation_guide"],
        "reflection_lm": used_config["reflection_lm"],
        "run_dir": str(run_dir),
        "gepa_best_iteration": optimization_summary["best_iteration"],
        "gepa_best_val_score": optimization_summary["best_score"],
        "best_val_score_trace": optimization_summary.get("best_val_score_trace"),
        "train_examples": optimization_summary["train_examples"],
        "val_examples": optimization_summary["val_examples"],
        "best_config_path": optimization_summary["best_candidate_path"],
        "post_run_config_path": str(Path(run.post_run_config_path)),
        "rerun_rollout_version": post_run_config["rollout_version"],
        "rerun_run_results_path": str(run_results_path),
        "rerun_eval_results_path": str(eval_results_path),
        "rerun_cost": run_cost_summary["overall"]["total_cost"],
        "rerun_cost_summary": run_cost_summary,
        "rerun_score": eval_summary["overall"]["avg_score"],
        "rerun_scores_by_rollout": eval_summary["overall"]["scores_by_rollout"],
        "rerun_train_score": eval_summary["train"]["avg_score"],
        "rerun_train_scores_by_rollout": eval_summary["train"]["scores_by_rollout"],
        "rerun_val_score": eval_summary["val"]["avg_score"],
        "rerun_val_scores_by_rollout": eval_summary["val"]["scores_by_rollout"],
    }


def build_post_gepa_json_row(summary: dict[str, Any]) -> dict[str, Any]:
    rerun_cost_summary = summary["rerun_cost_summary"]
    return {
        "task_id": summary["task_id"],
        "run_dir": summary["run_dir"],
        "model_name": summary["model_name"],
        "prompt_name": summary["prompt_name"],
        "seed": summary["seed"],
        "use_adaptation_guide": summary["use_adaptation_guide"],
        "reflection_lm": summary["reflection_lm"],
        "train_examples": summary["train_examples"],
        "val_examples": summary["val_examples"],
        "gepa": {
            "best_iteration": summary["gepa_best_iteration"],
            "best_val_score": summary["gepa_best_val_score"],
            "best_val_score_trace": summary["best_val_score_trace"],
            "best_config_path": summary["best_config_path"],
        },
        "rerun": {
            "rollout_version": summary["rerun_rollout_version"],
            "run_results_path": summary["rerun_run_results_path"],
            "eval_results_path": summary["rerun_eval_results_path"],
            "total": {
                "cost": rerun_cost_summary["overall"]["total_cost"],
                "avg_cost": rerun_cost_summary["overall"]["avg_cost"],
                "costs_by_rollout": rerun_cost_summary["overall"]["total_costs_by_rollout"],
                "avg_costs_by_rollout": rerun_cost_summary["overall"]["avg_costs_by_rollout"],
                "score": summary["rerun_score"],
                "scores_by_rollout": summary["rerun_scores_by_rollout"],
            },
            "train": {
                "cost": rerun_cost_summary["train"]["total_cost"],
                "avg_cost": rerun_cost_summary["train"]["avg_cost"],
                "costs_by_rollout": rerun_cost_summary["train"]["total_costs_by_rollout"],
                "avg_costs_by_rollout": rerun_cost_summary["train"]["avg_costs_by_rollout"],
                "score": summary["rerun_train_score"],
                "scores_by_rollout": summary["rerun_train_scores_by_rollout"],
            },
            "val": {
                "cost": rerun_cost_summary["val"]["total_cost"],
                "avg_cost": rerun_cost_summary["val"]["avg_cost"],
                "costs_by_rollout": rerun_cost_summary["val"]["total_costs_by_rollout"],
                "avg_costs_by_rollout": rerun_cost_summary["val"]["avg_costs_by_rollout"],
                "score": summary["rerun_val_score"],
                "scores_by_rollout": summary["rerun_val_scores_by_rollout"],
            },
        },
    }


def build_post_gepa_table_row(summary: dict[str, Any]) -> dict[str, str]:
    train_rollout_scores = summary["rerun_train_scores_by_rollout"]
    val_rollout_scores = summary["rerun_val_scores_by_rollout"]
    return {
        "Task": summary["task_id"],
        "Run": summary["run_dir"],
        "task_lm": summary["model_name"],
        "prompt_name": summary["prompt_name"],
        "seed": str(summary["seed"]),
        "use_adaptation": format_bool(summary["use_adaptation_guide"]),
        "reflection_lm": summary["reflection_lm"],
        "train_examples": str(summary["train_examples"]),
        "val_examples": str(summary["val_examples"]),
        "gepa_best_iteration": str(summary["gepa_best_iteration"]),
        "gepa_val_score": format_score(summary["gepa_best_val_score"]),
        "rerun_rollout_version": summary["rerun_rollout_version"],
        "rerun_cost": f"{summary['rerun_cost']:.4f}",
        "rerun_score": format_score(summary["rerun_score"]),
        "rerun_train_score": format_score(summary["rerun_train_score"]),
        "rerun_val_score": format_score(summary["rerun_val_score"]),
        "rerun_train_rollout_0": format_score(train_rollout_scores[0]),
        "rerun_train_rollout_1": format_score(train_rollout_scores[1]),
        "rerun_train_rollout_2": format_score(train_rollout_scores[2]),
        "rerun_val_rollout_0": format_score(val_rollout_scores[0]),
        "rerun_val_rollout_1": format_score(val_rollout_scores[1]),
        "rerun_val_rollout_2": format_score(val_rollout_scores[2]),
    }


def write_post_gepa_table(batch_dir: Path, pipeline_results: list[dict[str, Any]]) -> None:
    json_rows = []
    table_rows = []
    for result in pipeline_results:
        summary = result.get("post_gepa_summary")
        if not summary:
            continue
        json_rows.append(build_post_gepa_json_row(summary))
        table_rows.append(build_post_gepa_table_row(summary))

    json_rows.sort(key=lambda row: (row["task_id"], row["run_dir"]))
    table_rows.sort(key=lambda row: (row["Task"], row["Run"]))
    json_path = batch_dir / "post_gepa_results.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(json_rows, handle, indent=2)

    tsv_path = batch_dir / "post_gepa_results.tsv"
    csv_path = batch_dir / "post_gepa_results.csv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POST_GEPA_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(table_rows)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POST_GEPA_COLUMNS)
        writer.writeheader()
        writer.writerows(table_rows)


def run_full_pipeline(
    run: PreparedRun,
    console_lock: threading.Lock,
    launcher_log_path: Path,
    launcher_log_lock: threading.Lock,
) -> dict[str, Any]:
    optimize_command = [
        "uv",
        "run",
        "python",
        "-m",
        "src.gepa_optimize",
        "--config",
        run.optimize_config_path,
    ]
    optimize_return_code = run_logged_command(
        optimize_command,
        Path(run.optimize_log_path),
        f"run {run.index} optimize",
        console_lock,
        launcher_log_path,
        launcher_log_lock,
    )
    if optimize_return_code != 0:
        return {
            "index": run.index,
            "status": "failed_optimize",
            "return_code": optimize_return_code,
            "run_dir": run.run_dir,
        }

    post_run_config, post_run_config_path = create_post_gepa_run_config(run)
    collect_command = [
        "uv",
        "run",
        "python",
        "-m",
        "src.collect",
        "--config",
        str(post_run_config_path),
    ]
    collect_return_code = run_logged_command(
        collect_command,
        Path(run.post_collect_log_path),
        f"run {run.index} collect",
        console_lock,
        launcher_log_path,
        launcher_log_lock,
    )
    if collect_return_code != 0:
        return {
            "index": run.index,
            "status": "failed_collect",
            "return_code": collect_return_code,
            "run_dir": run.run_dir,
            "post_run_config_path": str(post_run_config_path),
        }

    evaluate_command = [
        "uv",
        "run",
        "python",
        "-m",
        "src.evaluate",
        "--config",
        str(post_run_config_path),
    ]
    evaluate_return_code = run_logged_command(
        evaluate_command,
        Path(run.post_evaluate_log_path),
        f"run {run.index} evaluate",
        console_lock,
        launcher_log_path,
        launcher_log_lock,
    )
    if evaluate_return_code != 0:
        return {
            "index": run.index,
            "status": "failed_evaluate",
            "return_code": evaluate_return_code,
            "run_dir": run.run_dir,
            "post_run_config_path": str(post_run_config_path),
        }

    post_gepa_summary = build_post_gepa_summary(run)
    with open(run.post_summary_path, "w", encoding="utf-8") as handle:
        json.dump(post_gepa_summary, handle, indent=2)

    return {
        "index": run.index,
        "status": "completed",
        "return_code": 0,
        "run_dir": run.run_dir,
        "post_run_config_path": str(post_run_config_path),
        "post_gepa_summary": post_gepa_summary,
    }


def launch_batch(batch_dir: Path, max_parallel: int) -> int:
    _, runs = read_batch(batch_dir)
    launcher_log_path = batch_dir / "launcher.log"
    launcher_log_lock = threading.Lock()
    console_lock = threading.Lock()

    print(f"Launching {len(runs)} run(s) from {batch_dir} with parallelism={max_parallel}")
    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [
            executor.submit(
                run_full_pipeline,
                run,
                console_lock,
                launcher_log_path,
                launcher_log_lock,
            )
            for run in runs
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda result: result["index"])
    with open(batch_dir / "launch_results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    write_post_gepa_table(batch_dir, results)

    failures = [result for result in results if result["return_code"] != 0]
    print()
    print("Launch summary:")
    for result in results:
        print(
            f"  idx={result['index']} status={result['status']} return_code={result['return_code']} "
            f"run_dir={result['run_dir']}"
        )

    return 1 if failures else 0


def find_first_best_hit(lines: list[str], best_score: float) -> BestHit:
    first_timestamp = None
    last_cost = None

    for line in lines:
        timestamp = parse_timestamp(line)
        if timestamp is not None and first_timestamp is None:
            first_timestamp = timestamp

        cost_match = RESULT_COST_RE.search(line)
        if cost_match:
            last_cost = float(cost_match.group("cost"))

        for pattern in RESULT_FIRST_BEST_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            score = float(match.group("score"))
            if not approximately_equal(score, best_score):
                continue
            if first_timestamp is None or timestamp is None or last_cost is None:
                raise ValueError("Could not resolve timestamp or cost for first-best hit")
            iteration = int(match.groupdict().get("iter") or 0)
            time_minutes = int((timestamp - first_timestamp).total_seconds() // 60)
            return BestHit(iteration=iteration, cost=last_cost, time_minutes=time_minutes)

    raise ValueError(f"Could not find first-best hit for score {best_score}")


def parse_result_table_row(log_path: Path) -> dict[str, str]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    summary = strict_load_json(log_path.parent / "shared" / "config" / "optimization_summary.json")
    used_config = strict_load_yaml(log_path.parent / "shared" / "config" / "used_config.yaml")

    header_match = next(
        (RESULT_HEADER_RE.search(line) for line in lines if RESULT_HEADER_RE.search(line)),
        None,
    )
    dataset_match = next(
        (RESULT_DATASET_RE.search(line) for line in lines if RESULT_DATASET_RE.search(line)),
        None,
    )
    initial_score_match = next(
        (RESULT_INITIAL_SCORE_RE.search(line) for line in lines if RESULT_INITIAL_SCORE_RE.search(line)),
        None,
    )

    if header_match is None:
        raise ValueError(f"Missing Task/Model/Prompt header in {log_path}")
    if dataset_match is None:
        raise ValueError(f"Missing dataset line in {log_path}")
    if initial_score_match is None:
        raise ValueError(f"Missing initial val score in {log_path}")

    best_score = float(summary["best_score"])
    best_hit = find_first_best_hit(lines, best_score)

    return {
        "Task": header_match.group("task"),
        "Run": log_path.as_posix(),
        "task_lm": header_match.group("model"),
        "N": dataset_match.group("count"),
        "budget": format_budget(float(used_config["max_cost"])),
        "use_adaptation": format_bool(bool(used_config["use_adaptation_guide"])),
        "reflection_lm": str(used_config["reflection_lm"]),
        "initial_val_score": format_score(float(initial_score_match.group("score"))),
        "best_val_score": format_score(best_score),
        "iter": str(best_hit.iteration),
        "cost": f"{best_hit.cost:.4f}",
        "time_mins": str(best_hit.time_minutes),
    }


def iter_log_paths(results_root: Path) -> list[Path]:
    return sorted(results_root.glob("**/gepa/*/gepa.log"))


def write_result_table(rows: list[dict[str, str]], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tsv_path = output_prefix.with_suffix(".tsv")
    csv_path = output_prefix.with_suffix(".csv")

    with open(tsv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_TABLE_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def prepare_command(args: argparse.Namespace) -> int:
    manifest_text = load_manifest_text(args.manifest)
    records = parse_manifest_table(manifest_text)
    batch_dir = build_batch_dir(args.batch_dir)
    runs = build_prepared_runs(manifest_text, records, batch_dir, args.max_parallel)
    print_prepared_runs(runs, batch_dir, args.max_parallel)
    print()
    print(f"Batch metadata: {batch_dir / 'batch.json'}")
    return 0


def launch_command(args: argparse.Namespace) -> int:
    batch_dir = Path(args.batch_dir)
    metadata, runs = read_batch(batch_dir)
    parallelism = args.max_parallel or metadata["max_parallel"]
    print_prepared_runs(runs, batch_dir, parallelism)
    if not args.yes and not prompt_for_confirmation("Launch these runs now?"):
        print("Launch cancelled.")
        return 0
    return launch_batch(batch_dir, parallelism)


def run_command(args: argparse.Namespace) -> int:
    manifest_text = load_manifest_text(args.manifest)
    records = parse_manifest_table(manifest_text)
    batch_dir = build_batch_dir(args.batch_dir)
    runs = build_prepared_runs(manifest_text, records, batch_dir, args.max_parallel)
    print_prepared_runs(runs, batch_dir, args.max_parallel)
    if not args.yes and not prompt_for_confirmation("Launch these runs now?"):
        print(f"Prepared batch kept at {batch_dir}")
        return 0
    return launch_batch(batch_dir, args.max_parallel)


def refresh_table_command(args: argparse.Namespace) -> int:
    rows = []
    skipped = []
    for log_path in iter_log_paths(Path(args.results_root)):
        try:
            row = parse_result_table_row(log_path)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append((str(log_path), str(exc)))
            continue

        if args.task and row["Task"] != args.task:
            continue
        if args.model and row["task_lm"] != args.model:
            continue
        if args.run_substring and args.run_substring not in row["Run"]:
            continue
        rows.append(row)

    rows.sort(key=lambda row: (row["Task"], row["Run"]))
    write_result_table(rows, Path(args.output_prefix))

    print(f"Wrote {len(rows)} rows to {Path(args.output_prefix).with_suffix('.tsv')}")
    print(f"Wrote {len(rows)} rows to {Path(args.output_prefix).with_suffix('.csv')}")
    if skipped:
        print(f"Skipped {len(skipped)} run(s):")
        for path, reason in skipped[:20]:
            print(f"  {path}: {reason}")
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Single deterministic CLI for preparing GEPA batches, launching optimization "
            "plus post-GEPA reruns, and refreshing GEPA result tables."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Generate GEPA configs and batch metadata only")
    prepare_parser.add_argument("--manifest", type=str, default="-", help="Manifest table path or '-' for stdin")
    prepare_parser.add_argument("--batch-dir", type=str, default=None, help="Batch directory to create")
    prepare_parser.add_argument("--max-parallel", type=int, default=4, help="Maximum parallel pipelines")
    prepare_parser.set_defaults(func=prepare_command)

    launch_parser = subparsers.add_parser("launch", help="Launch a previously prepared batch")
    launch_parser.add_argument("--batch-dir", type=str, required=True, help="Prepared batch directory")
    launch_parser.add_argument("--max-parallel", type=int, default=None, help="Override prepared parallelism")
    launch_parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    launch_parser.set_defaults(func=launch_command)

    run_parser = subparsers.add_parser("run", help="Prepare a batch, confirm, then launch")
    run_parser.add_argument("--manifest", type=str, default="-", help="Manifest table path or '-' for stdin")
    run_parser.add_argument("--batch-dir", type=str, default=None, help="Batch directory to create")
    run_parser.add_argument("--max-parallel", type=int, default=4, help="Maximum parallel pipelines")
    run_parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    run_parser.set_defaults(func=run_command)

    refresh_parser = subparsers.add_parser("refresh-table", help="Refresh GEPA result TSV/CSV tables")
    refresh_parser.add_argument("--results-root", type=str, default="results", help="Root directory to scan")
    refresh_parser.add_argument(
        "--output-prefix",
        type=str,
        default="analysis_results/gepa_runs",
        help="Prefix for output TSV and CSV files",
    )
    refresh_parser.add_argument("--task", type=str, default=None, help="Optional task filter")
    refresh_parser.add_argument("--model", type=str, default=None, help="Optional model filter")
    refresh_parser.add_argument(
        "--run-substring",
        type=str,
        default=None,
        help="Optional substring filter on run path",
    )
    refresh_parser.set_defaults(func=refresh_table_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
