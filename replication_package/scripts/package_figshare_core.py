import argparse
import json
import zipfile
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_OUTPUT = PACKAGE_DIR / "outputs" / "migration_analysis_figshare_core.zip"

CORE_FILES = [
    "README.md",
    "replication_package/RUNME.txt",
    "replication_package/pyproject.toml",
    "replication_package/uv.lock",
    "replication_package/data/curated_run_eval_inputs.json",
    "replication_package/data/manual_labels_raw.txt",
    "replication_package/data/model_intelligence_index.json",
    "replication_package/data/task_template_scores.json",
    "replication_package/outputs/data/manual_labels.json",
    "replication_package/outputs/data/included_task_model_seed_triplets.csv",
    "replication_package/outputs/data/included_task_model_seed_triplets.json",
    "replication_package/outputs/data/selected_test_rows.json",
    "replication_package/outputs/data/summary.json",
    "replication_package/outputs/data/tableau_data.json",
    "replication_package/outputs/data/task_diversity_metrics.json",
    "replication_package/outputs/figures/cost_performance_quadrants.png",
    "replication_package/outputs/figures/intelligence_vs_performance.png",
    "replication_package/outputs/figures/trajectory_diversity_side_by_side.png",
    "replication_package/outputs/figures/trajectory_diversity_side_by_side_labeled.png",
    "replication_package/outputs/tables/results_table.md",
    "replication_package/outputs/tables/results_table.tex",
]
GEPA_JSON_FILES = [
    "shared/config/optimization_summary.json",
    "shared/config/used_config.yaml",
    "shared/cost_summary.json",
]
SCRIPT_FILES = [
    "normalize_manual_labels.py",
    "build_triplet_coverage.py",
    "build_curated_run_eval_inputs.py",
    "package_figshare_core.py",
    "package_run_eval_artifacts.py",
    "plot_cost_performance_quadrants.py",
    "plot_intelligence_performance.py",
    "plot_trajectory_diversity_score.py",
    "render_from_intermediates.py",
    "render_table.py",
]
PROVENANCE_SCRIPT_FILES = [
    "collect_run_data.py",
    "compute_task_diversity.py",
    "derive_tableau_data.py",
    "render_all.py",
]


def add_file(files: dict[str, Path], logical_path: str | Path) -> None:
    logical = Path(logical_path)
    actual = REPO_DIR / logical
    if actual.exists() and actual.is_file():
        files[logical.as_posix()] = actual


def add_scripts(files: dict[str, Path]) -> None:
    for script_name in [*SCRIPT_FILES, *PROVENANCE_SCRIPT_FILES]:
        add_file(files, Path("replication_package") / "scripts" / script_name)


def add_best_configs_and_summaries(files: dict[str, Path]) -> None:
    summary_path = PACKAGE_DIR / "outputs" / "data" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["rows"]:
        agent_file = row.get("agent_file")
        if not agent_file:
            continue
        agent_path = Path(agent_file)
        add_file(files, agent_path)
        gepa_dir = agent_path.parents[2]
        for relative_path in GEPA_JSON_FILES:
            add_file(files, gepa_dir / relative_path)


def write_readme(files: dict[str, Path], output: Path) -> Path:
    instructions_path = output.with_suffix(".instructions.txt")
    instructions = f"""Migration Analysis Core Replication Artifact

Quick rerun:

1. Decompress the compact archive.

  unzip {output.name}

2. Recreate the table and figures from the included intermediate data.

  uv run --project replication_package python replication_package/scripts/render_from_intermediates.py

3. Inspect the regenerated outputs.

  replication_package/outputs/tables/results_table.tex
  replication_package/outputs/tables/results_table.md
  replication_package/outputs/figures/cost_performance_quadrants.png
  replication_package/outputs/figures/intelligence_vs_performance.png
  replication_package/outputs/figures/trajectory_diversity_side_by_side.png
  replication_package/outputs/figures/trajectory_diversity_side_by_side_labeled.png

The compact inputs are in replication_package/outputs/data/*.json. To recompute
summary.json and tableau_data.json, also decompress
migration_analysis_run_eval_artifacts.zip and follow README.md.
"""
    instructions_path.write_text(instructions, encoding="utf-8")
    files[instructions_path.name] = instructions_path
    return instructions_path


def write_zip(files: dict[str, Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    instructions_path = write_readme(files, output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "file_count": len(files),
        "files": sorted(files),
        "omitted": [
            "raw trace_*.json files",
            "large collect/evaluate logs",
            "local virtual environments and __pycache__",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files[manifest_path.name] = manifest_path

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for archive_path in sorted(files):
            archive.write(files[archive_path], archive_path)

    print(output)
    print(manifest_path)
    print(instructions_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    files: dict[str, Path] = {}
    for logical_path in CORE_FILES:
        add_file(files, logical_path)
    add_scripts(files)
    add_best_configs_and_summaries(files)

    total_size = sum(path.stat().st_size for path in files.values())
    print(f"Collected {len(files)} files ({total_size / 1024 ** 2:.1f} MiB before compression)")
    write_zip(files, args.output)


if __name__ == "__main__":
    main()
