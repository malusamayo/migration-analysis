# Better Harnesses, Smaller Models: Building 90% Cheaper Agents via Automated Harness Adaptation

This repository contains the replication package for the paper result table and figures. 

## Download Artifacts

The large replication artifacts are hosted on figshare:

```text
https://figshare.com/s/520e6259e3cc730c358d
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

## Run One Experiment From Raw Data

The committed root `data/*.json` files are the task datasets. The experiment
runner reads the task's `tasks/<task_id>/run*.yaml` config, which points to the
corresponding raw data file through `data_path`.

Example: run a small baseline experiment on `data/machine_operating_s2l.json`.

1. Prepare and launch one baseline run.

```bash
uv run python run.py run-baseline \
  --manifest - \
  --batch-dir generated/baseline_batches/example_machine_operating \
  --yes <<'EOF'
task_id	model_name	max_examples	n_responses
machine_operating_s2l	gemma-4-26b-a4b	10	1
EOF
```

2. Inspect the generated batch metadata and outputs.

```text
generated/baseline_batches/example_machine_operating/batch.json
generated/baseline_batches/example_machine_operating/configs/
generated/baseline_batches/example_machine_operating/logs/
results/machine_operating_s2l/gemma-4-26b-a4b_default/rollouts/baseline/
```

For paper-scale baseline runs, use `max_examples=100` and `n_responses=3`
for 100-example tasks. WebArena uses 50 examples in this package. The task
server Docker image and model/API credentials must be available before launching
new experiments.

Example: run a small GEPA optimization on the same raw task data.

1. Prepare and launch one optimization run.

```bash
uv run python run.py run \
  --manifest - \
  --batch-dir generated/gepa_batches/example_machine_operating \
  --max-parallel 1 \
  --yes <<'EOF'
Task	task_lm	N	budget ($)	use_adaptation	reflection_lm	num_exploration	seed
machine_operating_s2l	gemma-4-26b-a4b	10	2	TRUE	gemini-3.1-pro-preview	1	0
EOF
```

2. Inspect the generated optimization metadata and outputs.

```text
generated/gepa_batches/example_machine_operating/batch.json
generated/gepa_batches/example_machine_operating/configs/
generated/gepa_batches/example_machine_operating/logs/
results/machine_operating_s2l/gemma-4-26b-a4b_default/gepa/seed0/
results/machine_operating_s2l/gemma-4-26b-a4b_default/rollouts/gepa_seed0_best_best_config/
```

The optimization command first runs GEPA on the train/validation split, then
runs the selected best harness on the full task dataset. For paper-scale
optimization, use the task/model/seed rows listed in
`replication_package/outputs/data/included_task_model_seed_triplets.csv`, with
the full training budget used by the paper rather than the small `budget ($)=2`
smoke-test value above.

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
