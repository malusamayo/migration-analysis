# Better Harnesses, Smaller Models: Building 90% Cheaper Agents via Automated Harness Adaptation

This repository contains the replication package for the paper result table and figures. 

## Download Artifacts

The large replication artifacts are hosted on figshare:

```text
FIGSHARE_LINK_OR_DOI_HERE
```

Download these files into the repository root:

```text
migration_analysis_figshare_core.zip
migration_analysis_run_eval_artifacts.zip
```

If a checksum file is provided, download it into the repository root and verify
the archives:

```bash
sha256sum -c SHA256SUMS.txt
```

The compact archive is sufficient to rerender the paper table and figures. The
run/eval archive is only needed to recompute `summary.json`,
`selected_test_rows.json`, and `tableau_data.json` from the curated
`run.json`/`eval_results.yaml` files.

## Quick Rerun

1. Decompress the compact archive.

```bash
unzip migration_analysis_figshare_core.zip
```

2. Recreate the table and figures from the included intermediate data.

```bash
uv run --project replication_package python replication_package/scripts/render_from_intermediates.py
```

3. Inspect the regenerated outputs.

```text
replication_package/outputs/tables/results_table.tex
replication_package/outputs/tables/results_table.md
replication_package/outputs/figures/cost_performance_quadrants.png
replication_package/outputs/figures/intelligence_vs_performance.png
replication_package/outputs/figures/trajectory_diversity_side_by_side.png
replication_package/outputs/figures/trajectory_diversity_side_by_side_labeled.png
```

## Run/Eval Rerun

Use this path to recompute `summary.json`, `selected_test_rows.json`, and
`tableau_data.json` from the curated `run.json` and `eval_results.yaml` files.
The curated run list is stored in
`replication_package/data/curated_run_eval_inputs.json`.

1. Decompress both archives.

```bash
unzip migration_analysis_figshare_core.zip
unzip migration_analysis_run_eval_artifacts.zip
```

2. Recompute the run-level summary.

```bash
uv run --project replication_package python replication_package/scripts/collect_run_data.py
```

3. Recompute the selected rows and table/figure input data.

```bash
uv run --project replication_package python replication_package/scripts/derive_tableau_data.py
uv run --project replication_package python replication_package/scripts/build_triplet_coverage.py
```

4. Regenerate the final table and figures.

```bash
uv run --project replication_package python replication_package/scripts/render_from_intermediates.py
```

The run/eval archive contains the files needed to recompute the table inputs. It
does not include raw trace files, so the diversity figure uses the included
`task_diversity_metrics.json`.

## Data Flow

1. `collect_run_data.py` reads `curated_run_eval_inputs.json`, the curated
   `run.json` files, and the curated `eval_results.yaml` files, then writes
   `replication_package/outputs/data/summary.json`.
2. `derive_tableau_data.py` reads `summary.json`, applies the validation
   selection used in the paper, and writes `selected_test_rows.json` and
   `tableau_data.json`.
3. `render_table.py` reads `summary.json` and writes the LaTeX and Markdown
   result tables.
4. `plot_cost_performance_quadrants.py` reads `tableau_data.json` and writes the
   cost/performance figure.
5. `plot_intelligence_performance.py` reads `tableau_data.json` and
   `model_intelligence_index.json`, then writes the intelligence/performance
   figure.
6. `plot_trajectory_diversity_score.py` reads `tableau_data.json`,
   `task_diversity_metrics.json`, and `task_template_scores.json`, then writes
   the diversity figures.
7. `normalize_manual_labels.py` reads `manual_labels_raw.txt` and writes
   normalized failure-mode labels.
8. `build_triplet_coverage.py` reads `summary.json` and writes the
   task/model/seed coverage table.

## Included Data

```text
replication_package/data/curated_run_eval_inputs.json
replication_package/data/manual_labels_raw.txt
replication_package/data/model_intelligence_index.json
replication_package/data/task_template_scores.json
replication_package/outputs/data/summary.json
replication_package/outputs/data/selected_test_rows.json
replication_package/outputs/data/tableau_data.json
replication_package/outputs/data/task_diversity_metrics.json
replication_package/outputs/data/manual_labels.json
replication_package/outputs/data/included_task_model_seed_triplets.csv
```
