import argparse
import json
import zipfile
from pathlib import Path

from build_triplet_coverage import build_curated_run_eval_rows
from build_triplet_coverage import build_triplet_rows
from build_triplet_coverage import load_summary


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PACKAGE_DIR.parent
DEFAULT_OUTPUT = PACKAGE_DIR / "outputs" / "migration_analysis_run_eval_artifacts.zip"


def add_file(files: dict[str, Path], logical_path: Path, missing: list[str]) -> None:
    actual_path = REPO_DIR / logical_path
    if actual_path.exists() and actual_path.is_file():
        files[logical_path.as_posix()] = actual_path
    else:
        missing.append(logical_path.as_posix())


def collect_run_eval_files(summary_json: Path) -> tuple[dict[str, Path], list[str], dict[str, int]]:
    files: dict[str, Path] = {}
    missing: list[str] = []
    summary_rows = load_summary(summary_json)
    curated_rows = build_curated_run_eval_rows(summary_rows)
    triplet_rows = build_triplet_rows(summary_rows)
    group_counts: dict[str, int] = {}
    for triplet in triplet_rows:
        if triplet["included_in_curated_run_eval_archive"]:
            group_counts[triplet["coverage_group"]] = group_counts.get(triplet["coverage_group"], 0) + 1

    for row in curated_rows:
        add_file(files, Path(row["linkage"]["run_results_path"]), missing)
        add_file(files, Path(row["linkage"]["eval_results_path"]), missing)

    counts = {
        "run_config_count": len(curated_rows),
        "baseline_run_config_count": sum(row["rollout_version"].startswith("baseline") for row in curated_rows),
        "optimized_run_config_count": sum(not row["rollout_version"].startswith("baseline") for row in curated_rows),
        **group_counts,
    }
    return files, missing, counts


def write_zip(files: dict[str, Path], output: Path, counts: dict[str, int]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "file_count": len(files),
        "counts": counts,
        "files": sorted(files),
        "notes": [
            "Path-preserving archive of curated run.json and eval_results.yaml files.",
            "The included optimized triplets are listed in replication_package/outputs/data/included_task_model_seed_triplets.csv.",
            "Unzip at repository root alongside the compact figshare package.",
            "This archive does not include trace_*.json files or collect/evaluate logs.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(manifest_path, manifest_path.name)
        for index, archive_path in enumerate(sorted(files), start=1):
            if index == 1 or index % 25 == 0 or index == len(files):
                print(f"Adding {index}/{len(files)}: {archive_path}", flush=True)
            archive.write(files[archive_path], archive_path)

    print(output)
    print(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=PACKAGE_DIR / "outputs" / "data" / "summary.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    files, missing, counts = collect_run_eval_files(args.summary_json)
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing[:20]}")

    total_size = sum(path.stat().st_size for path in files.values())
    print(f"Collected {len(files)} files ({total_size / 1024 ** 3:.2f} GiB before compression)")
    print(json.dumps(counts, sort_keys=True))
    write_zip(files, args.output, counts)


if __name__ == "__main__":
    main()
