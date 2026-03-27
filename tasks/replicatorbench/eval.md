Reads post_registration.json from the workspace and scores it against expected_post_registration.json in the log_dir groundtruth folder.

The agent must reproduce results computationally (by translating and running the replication scripts), NOT by reading the paper — the paper is withheld from the workspace.

Scoring is fully deterministic — no LLM judge:

1. sample_size: parse the first integer from the string. Exact match required.
2. For each numerical_result in the ground truth, find the best-matching result in the agent output and check:
   - value: 2% relative tolerance
   - confidence_interval.lower / upper: 2% relative tolerance
   - direction: exact match (case-insensitive)
   - statistical_significance: boolean equality (normalises "true"/1/true)

Fields annotated "not stated" in the ground truth are skipped.
Final score = mean of all per-field checks across all numerical results.
If both expected_post_registration.json and expected_post_registration_2.json are present, the higher score is kept.
