import re


_COST_RE = re.compile(r"\[cost\] total=\$(?P<cost>\d+(?:\.\d+)?)")
_ITERATION_RE = re.compile(r"Iteration (?P<iter>\d+):")
_BASE_VAL_SCORE_RE = re.compile(
    r"Iteration 0: Base program full valset score: (?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
_BEST_VAL_SCORE_RE = re.compile(
    r"Iteration (?P<iter>\d+): Best score on valset: (?P<score>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def extract_best_val_score_trace(log_path: str) -> list[dict[str, float | int]]:
    with open(log_path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    current_cost = None
    current_iteration = None
    explicit_best_scores = {}
    iteration_costs = {}
    seen_iterations = set()
    for line in lines:
        iteration_match = _ITERATION_RE.search(line)
        if iteration_match:
            current_iteration = int(iteration_match.group("iter"))
            seen_iterations.add(current_iteration)
            if current_iteration == 0 and current_cost is not None and 0 not in iteration_costs:
                iteration_costs[0] = current_cost

        cost_match = _COST_RE.search(line)
        if cost_match:
            current_cost = float(cost_match.group("cost"))
            if current_iteration is not None:
                iteration_costs[current_iteration] = current_cost

        base_match = _BASE_VAL_SCORE_RE.search(line)
        if base_match:
            if current_cost is None:
                raise ValueError(f"Missing cumulative cost before base val score in {log_path}")
            seen_iterations.add(0)
            iteration_costs[0] = current_cost
            explicit_best_scores[0] = float(base_match.group("score"))
            continue

        best_match = _BEST_VAL_SCORE_RE.search(line)
        if best_match:
            iteration = int(best_match.group("iter"))
            seen_iterations.add(iteration)
            if current_cost is not None and iteration not in iteration_costs:
                iteration_costs[iteration] = current_cost
            explicit_best_scores[iteration] = float(best_match.group("score"))

    if not seen_iterations:
        raise ValueError(f"Could not extract iteration trace from {log_path}")
    if 0 not in explicit_best_scores:
        raise ValueError(f"Missing base val score in {log_path}")

    trace = []
    best_val_score = None
    last_cost = None
    for iteration in range(max(seen_iterations) + 1):
        if iteration in explicit_best_scores:
            best_val_score = explicit_best_scores[iteration]
        if best_val_score is None:
            raise ValueError(f"Missing best val score by iteration {iteration} in {log_path}")
        if iteration in iteration_costs:
            last_cost = iteration_costs[iteration]
        if last_cost is None:
            raise ValueError(f"Missing cumulative cost by iteration {iteration} in {log_path}")
        trace.append(
            {
                "iteration": iteration,
                "cost": last_cost,
                "best_val_score": best_val_score,
            }
        )

    return trace
