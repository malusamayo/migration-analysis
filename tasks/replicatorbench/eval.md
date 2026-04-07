Reads staged JSON artifacts from the workspace and scores the task in four
stages:

1. Extraction
   - compares `extraction.json` against `expected_post_registration*.json`
   - uses deterministic field checks on claim/data/method/result fields
2. Design
   - compares `design.json` against structured fields derived from the same
     expected post-registration references
3. Execution
   - checks process artifacts in `execution_summary.json`
   - scores executed result reproduction from `post_registration.json`
4. Interpretation
   - checks whether `final_report.json` is internally consistent with the
     executed result and whether the support decision is justified

The evaluator also reports `result_score`, a legacy reproduction-only subscore
for sample size and numerical-result fields. Multiple ground-truth variants are
supported by keeping the best-scoring variant.
