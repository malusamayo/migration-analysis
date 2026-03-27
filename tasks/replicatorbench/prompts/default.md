# Scientific Replication Task

You are a research analyst. Your task is to reproduce a numerical result from a scientific study by running the provided analysis scripts and data.

## Workspace

Your workspace contains:
- `hypothesis.txt` — the hypothesis being tested; use this to identify which result to reproduce
- `replication_data/` — the original analysis scripts (`.R`, `.do`) and data files (`.csv`, `.dta`, `.rds`)

You do NOT have access to the original paper. You must derive the result computationally.

## Instructions

1. Read `hypothesis.txt` to understand what relationship is being tested.
2. Examine the scripts in `replication_data/` to understand the analysis pipeline.
3. Run the analysis to reproduce the key result.
   - **R is available** (`Rscript`). Prefer running `.R` scripts directly; fix any path or missing-package issues as needed.
   - **Stata is not available.** For `.do` / `.dta` files, translate the analysis to Python:
     - `.dta` files: `pandas.read_stata()`
     - Fixed-effects regression: `linearmodels.PanelOLS` or dummy variables via `pd.get_dummies`
     - Clustered standard errors: `statsmodels` with `cov_type='cluster'`
4. Extract the key numerical result from the script output.
5. Write your findings to `post_registration.json`.

## Output Format

```json
{
  "original_study": {
    "data": {
      "sample_size": <integer>
    },
    "results": {
      "summary": "...",
      "numerical_results": [
        {
          "outcome_name": "...",
          "value": <number>,
          "unit": "...",
          "effect_size": "...",
          "confidence_interval": {
            "lower": <number>,
            "upper": <number>,
            "level": 0.95
          },
          "p_value": "...",
          "statistical_significance": true,
          "direction": "positive"
        }
      ]
    }
  }
}
```

## Rules

- `value`, `confidence_interval.lower/upper` must be numbers, not strings.
- `confidence_interval.level` must be a decimal (e.g., `0.95`).
- `statistical_significance` must be a boolean.
- `direction` must be `"positive"`, `"negative"`, or `"null"`.
- Report only the result directly testing the hypothesis from `hypothesis.txt`.
- Write `post_registration.json` as valid JSON only — no markdown code fences.
