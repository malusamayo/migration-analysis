import json
import os
import shutil
from dataclasses import asdict, dataclass
from typing import Any

from .common import hash_text


@dataclass
class CandidateRecord:
    """Complete lifecycle record for one candidate in the optimization."""

    # Identity
    iteration: int
    code_hash: str
    agent_code: str

    # Lineage
    parent_hash: str | None
    code_diff: str | None
    components_updated: list[str]

    # Reflection phase context (parent's scores on the subsample that motivated this proposal)
    reflection_scores: list[float] | None = None
    reflection_score_sum: float | None = None
    reflection_trajectory_dir: str | None = None
    reflection_batch_prompts: list[str] | None = None

    # Proposal evaluation (THIS candidate's scores on same subsample)
    subsample_scores: list[float] | None = None
    subsample_score_sum: float | None = None
    subsample_outputs: list[dict[str, Any]] | None = None
    subsample_trajectory_dir: str | None = None
    score_delta: float | None = None
    status: str | None = None  # "seed" | "accepted_on_subsample" | "rejected_on_subsample"
    per_example_feedback: list[dict[str, Any]] | None = None

    # Validation (only if accepted)
    val_score: float | None = None
    val_per_example_feedback: list[dict[str, Any]] | None = None
    val_trajectory_dir: str | None = None

    @property
    def subsample_avg(self) -> float | None:
        if self.subsample_scores is None:
            return None
        return sum(self.subsample_scores) / len(self.subsample_scores)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CandidateRecord":
        return cls(**d)


def _sanitize_feedback(text: str) -> str:
    """Sanitize feedback text for markdown table display."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _append_feedback_table(
    lines: list[str],
    heading: str,
    feedback_rows: list[dict[str, Any]],
    before_scores: list[float] | None = None,
) -> None:
    """Append a markdown feedback table to the provided line buffer."""
    if not feedback_rows:
        return

    lines.append(heading)
    lines.append("")
    if before_scores and len(before_scores) == len(feedback_rows):
        lines.append("| Ex | Before | After | Feedback |")
        lines.append("|----|--------|-------|----------|")
        for i, (fb, before) in enumerate(zip(feedback_rows, before_scores)):
            feedback_text = _sanitize_feedback(fb.get("feedback", ""))
            lines.append(
                f"| {i} | {before} | {fb.get('score', '-')} | {feedback_text} |"
            )
    else:
        lines.append("| Ex | Score | Feedback |")
        lines.append("|----|-------|----------|")
        for i, fb in enumerate(feedback_rows):
            feedback_text = _sanitize_feedback(fb.get("feedback", ""))
            lines.append(f"| {i} | {fb.get('score', '-')} | {feedback_text} |")
    lines.append("")


class ProposerMemory:
    """Centralized memory for the proposer's optimization history."""

    def __init__(self, run_dir: str):
        self._run_dir = run_dir
        self._memory_path = os.path.join(run_dir, "shared", "proposal_memory", "memory.json")
        self._records: list[CandidateRecord] = []
        self._by_hash: dict[str, CandidateRecord] = {}
        self._pending_reflection: dict[str, Any] | None = None
        # Transient: current reflective dataset (trajectories + eval details).
        # Not persisted — too large and reconstructable from disk artifacts.
        self._reflective_records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self._memory_path):
            return
        with open(self._memory_path) as f:
            data = json.load(f)
        for entry in data:
            record = CandidateRecord.from_dict(entry)
            self._records.append(record)
            self._by_hash[record.code_hash] = record

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._memory_path), exist_ok=True)
        with open(self._memory_path, "w") as f:
            json.dump([r.to_dict() for r in self._records], f, indent=2, default=str)

    # ---- Recording methods ----

    def record_seed(
        self,
        agent_code: str,
        scores: list[float],
        outputs: list[dict[str, Any]],
        batch: list[dict[str, Any]],
        trajectory_dir: str | None = None,
    ) -> None:
        """Record the initial seed candidate and its evaluation scores.

        The seed is evaluated on the full dataset (not a subsample), so the
        average score is recorded as val_score.
        """
        avg = sum(scores) / len(scores) if scores else 0.0
        feedback = [
            {
                "prompt": ex.get("prompt", "")[:300],
                "score": s,
                "feedback": o.get("feedback", "") if isinstance(o, dict) else "",
            }
            for ex, s, o in zip(batch, scores, outputs)
        ]
        # Seed is evaluated on the full dataset — that IS the validation eval.
        record = CandidateRecord(
            iteration=0,
            code_hash=hash_text(agent_code),
            agent_code=agent_code,
            parent_hash=None,
            code_diff=None,
            components_updated=[],
            status="seed",
            val_score=avg,
            val_per_example_feedback=feedback,
            val_trajectory_dir=trajectory_dir,
        )
        self._records.append(record)
        self._by_hash[record.code_hash] = record
        self.save()

    def record_reflection(
        self,
        iteration: int,
        parent_code: str,
        scores: list[float],
        batch_prompts: list[str],
        trajectory_dir: str,
    ) -> None:
        """Record reflection phase results. Stores context for the upcoming proposal."""
        self._pending_reflection = {
            "iteration": iteration,
            "parent_hash": hash_text(parent_code),
            "reflection_scores": list(scores),
            "reflection_score_sum": sum(scores),
            "reflection_batch_prompts": list(batch_prompts),
            "reflection_trajectory_dir": trajectory_dir,
        }

    def record_proposal(
        self,
        iteration: int,
        parent_code: str,
        proposed_code: str,
        components_updated: list[str],
        code_diff: str,
    ) -> None:
        """Record a new proposal. Attaches pending reflection context."""
        reflection = self._pending_reflection

        record = CandidateRecord(
            iteration=iteration,
            code_hash=hash_text(proposed_code),
            agent_code=proposed_code,
            parent_hash=hash_text(parent_code),
            code_diff=code_diff,
            components_updated=list(components_updated),
        )

        if reflection and reflection["parent_hash"] == record.parent_hash:
            record.reflection_scores = reflection["reflection_scores"]
            record.reflection_score_sum = reflection["reflection_score_sum"]
            record.reflection_batch_prompts = reflection["reflection_batch_prompts"]
            record.reflection_trajectory_dir = reflection["reflection_trajectory_dir"]

        self._records.append(record)
        self._by_hash[record.code_hash] = record
        self.save()

    def record_outcome(
        self,
        iteration: int,
        proposed_code: str,
        scores: list[float],
        outputs: list[dict[str, Any]],
        batch: list[dict[str, Any]],
        trajectory_dir: str | None = None,
    ) -> None:
        """Record subsample evaluation outcome for a proposed candidate."""
        proposed_hash = hash_text(proposed_code)
        record = self._by_hash.get(proposed_hash)
        if record is None:
            raise KeyError(
                f"record_outcome: no record found for hash {proposed_hash} "
                f"(iteration {iteration}). Was record_proposal called first?"
            )

        score_list = list(scores)
        after_sum = sum(score_list)
        record.subsample_scores = score_list
        record.subsample_score_sum = after_sum
        record.subsample_outputs = [o if isinstance(o, dict) else {"raw": o} for o in outputs]
        record.subsample_trajectory_dir = trajectory_dir

        before_sum = record.reflection_score_sum
        if before_sum is not None:
            record.score_delta = (after_sum - before_sum) / len(score_list) if score_list else 0.0
            record.status = (
                "accepted_on_subsample" if record.score_delta > 0 else "rejected_on_subsample"
            )
        else:
            record.status = "accepted_on_subsample" if after_sum > 0 else "rejected_on_subsample"

        record.per_example_feedback = [
            {
                "prompt": ex.get("prompt", "")[:300],
                "score": s,
                "feedback": o.get("feedback", "") if isinstance(o, dict) else "",
            }
            for ex, s, o in zip(batch, scores, outputs)
        ]

        self.save()

    def record_val_score(
        self,
        code_hash: str,
        val_score: float,
        scores: list[float],
        outputs: list[dict[str, Any]],
        batch: list[dict[str, Any]],
        trajectory_dir: str | None = None,
    ) -> None:
        """Record validation score for an accepted candidate."""
        record = self._by_hash.get(code_hash)
        if record is None:
            raise KeyError(
                f"record_val_score: no record found for hash {code_hash}. "
                "Was record_proposal + record_outcome called first?"
            )
        record.val_score = val_score
        record.val_per_example_feedback = [
            {
                "prompt": ex.get("prompt", "")[:300],
                "score": s,
                "feedback": o.get("feedback", "") if isinstance(o, dict) else "",
            }
            for ex, s, o in zip(batch, scores, outputs)
        ]
        record.val_trajectory_dir = trajectory_dir
        self.save()

    def record_reflective_dataset(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        """Store the current iteration's reflective dataset (trajectories + eval details).

        This is transient — not persisted to disk — because trajectory markdown
        is large and only needed for the upcoming proposal workspace.
        """
        self._reflective_records = list(records)

    # ---- Workspace staging ----

    def stage_workspace(self, workspace: str) -> None:
        """Write all memory-owned files into the proposer workspace under memory/.

        Layout:
        - memory/scoreboard.md — full optimization history
        - memory/current/overview.md — current candidate's code + eval results
        - memory/current/trajectories/ — current iteration's raw trajectory JSON
        - memory/past_agents.md — past candidate summaries with scores and agent code
        """
        mem_dir = os.path.join(workspace, "memory")
        if os.path.exists(mem_dir):
            shutil.rmtree(mem_dir)
        os.makedirs(mem_dir, exist_ok=True)

        # Scoreboard
        with open(os.path.join(mem_dir, "scoreboard.md"), "w") as f:
            f.write(self.format_scoreboard())

        # Current: overview.md + trajectories/ from reflective dataset
        current_dir = os.path.join(mem_dir, "current")
        os.makedirs(current_dir, exist_ok=True)
        current = self.get_current()
        if current:
            with open(os.path.join(current_dir, "overview.md"), "w") as f:
                f.write(self._format_current_overview(current))
        traj_dir = os.path.join(current_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        teacher_traj_dir = os.path.join(current_dir, "teacher_trajectories")
        os.makedirs(teacher_traj_dir, exist_ok=True)
        for i, record in enumerate(self._reflective_records):
            trajectory_json_path = record.get("Agent Trajectory JSON Path")
            if trajectory_json_path:
                shutil.copy2(trajectory_json_path, os.path.join(traj_dir, f"example{i}.json"))
            teacher_trajectory_json_path = record.get("Teacher Trajectory JSON Path")
            if teacher_trajectory_json_path:
                shutil.copy2(
                    teacher_trajectory_json_path,
                    os.path.join(teacher_traj_dir, f"example{i}.json"),
                )

        with open(os.path.join(mem_dir, "past_agents.md"), "w") as f:
            f.write(self.format_past_agents())

    def _append_candidate_header(self, lines: list[str], r: CandidateRecord) -> None:
        lines.append(f"# Iteration {r.iteration} — `{r.code_hash}`")
        lines.append("")
        lines.append(f"- **Status:** {r.status}")
        if r.subsample_avg is not None:
            lines.append(f"- **Train avg:** {r.subsample_avg:.3f}")
        if r.score_delta is not None:
            lines.append(f"- **Delta:** {r.score_delta:+.3f}")
        if r.val_score is not None:
            lines.append(f"- **Val score:** {r.val_score:.3f}")
        if r.parent_hash:
            lines.append(f"- **Parent:** `{r.parent_hash}`")
        lines.append("")

    def _append_candidate_results(self, lines: list[str], r: CandidateRecord) -> None:
        if r.per_example_feedback:
            _append_feedback_table(
                lines,
                "## Evaluation Results",
                r.per_example_feedback,
                before_scores=r.reflection_scores,
            )
        if r.val_per_example_feedback:
            heading = (
                "## Validation Evaluation Results"
                if r.per_example_feedback
                else "## Evaluation Results"
            )
            _append_feedback_table(lines, heading, r.val_per_example_feedback)

    def _append_agent_code(self, lines: list[str], r: CandidateRecord) -> None:
        lines.append("## Agent Code")
        lines.append("```python")
        lines.append(r.agent_code)
        lines.append("```")
        lines.append("")

    def _format_candidate_overview(self, r: CandidateRecord) -> str:
        """Format overview.md for a single candidate: agent code + eval results."""
        lines: list[str] = []
        self._append_candidate_header(lines, r)
        self._append_candidate_results(lines, r)
        self._append_agent_code(lines, r)

        return "\n".join(lines)

    def _format_current_overview(self, r: CandidateRecord) -> str:
        """Format current overview as results, reflection table, then agent code."""
        lines: list[str] = []
        self._append_candidate_header(lines, r)
        self._append_candidate_results(lines, r)
        if self._reflective_records:
            lines.append(self._format_reflection_section().rstrip())
            lines.append("")
        self._append_agent_code(lines, r)

        return "\n".join(lines)

    def _format_reflection_section(self) -> str:
        """Format the current reflection results with links to trajectory files."""
        lines: list[str] = []
        lines.append("## Current Reflection Results")
        lines.append("")
        lines.append(
            "These are the results from running the current agent on the reflection subsample. "
            "Each example has a raw trajectory JSON file in `trajectories/`. "
            "If a teacher trajectory directory is configured, matched teacher traces are in "
            "`teacher_trajectories/`."
        )
        lines.append("")
        lines.append("| Ex | Score | Feedback | Agent Trajectory JSON | Teacher Trajectory JSON |")
        lines.append("|----|-------|----------|-----------------------|-------------------------|")
        for i, record in enumerate(self._reflective_records):
            eval_output = record.get("eval_output", {})
            score = eval_output.get("score", "-")
            feedback = _sanitize_feedback(record.get("Evaluation Feedback", ""))
            if record.get("Agent Trajectory JSON Path"):
                raw_link = f"[trajectories/example{i}.json](trajectories/example{i}.json)"
            else:
                raw_link = "-"
            if record.get("Teacher Trajectory JSON Path"):
                teacher_link = (
                    f"[teacher_trajectories/example{i}.json](teacher_trajectories/example{i}.json)"
                )
            else:
                teacher_link = "-"
            lines.append(f"| {i} | {score} | {feedback} | {raw_link} | {teacher_link} |")
        lines.append("")
        return "\n".join(lines)

    # ---- Query methods ----

    def get_current(self) -> CandidateRecord | None:
        """Return the current best candidate (last accepted or seed)."""
        for record in reversed(self._records):
            if record.status in ("seed", "accepted_on_subsample"):
                return record
        return None

    def get_all(self) -> list[CandidateRecord]:
        return list(self._records)

    def format_scoreboard(self) -> str:
        """Format a succinct optimization history table for the proposer workspace.

        Full past-candidate details are in memory/past_agents.md. The current
        candidate and its trajectories are in memory/current/.
        """
        current = self.get_current()
        current_hash = current.code_hash if current else None

        accepted_count = sum(
            1 for r in self._records if r.status in ("seed", "accepted_on_subsample")
        )
        rejected_count = sum(
            1 for r in self._records if r.status == "rejected_on_subsample"
        )
        best_val = max(
            (r.val_score for r in self._records if r.val_score is not None),
            default=None,
        )

        lines: list[str] = []

        # Header
        lines.append("# Optimization Scoreboard")
        lines.append("")
        best_val_str = f"{best_val:.3f}" if best_val is not None else "N/A"
        lines.append(f"- **Total candidates:** {len(self._records)} "
                      f"({accepted_count} accepted, {rejected_count} rejected)")
        if current_hash:
            lines.append(f"- **Current best:** `{current_hash}` (val: {best_val_str})")
        lines.append("")
        lines.append("See `memory/past_agents.md` for past candidate scores and agent code.")
        lines.append("")

        # Summary table
        lines.append("| Iter | Hash | Status | Train Avg | Delta | Val Score |")
        lines.append("|------|------|--------|-----------|-------|-----------|")
        for r in self._records:
            h = r.code_hash
            avg_str = f"{r.subsample_avg:.3f}" if r.subsample_avg is not None else "-"
            delta_str = f"{r.score_delta:+.3f}" if r.score_delta is not None else "-"
            val_str = f"{r.val_score:.3f}" if r.val_score is not None else "-"
            status = r.status or "pending"
            current_marker = " ★" if r.code_hash == current_hash else ""
            lines.append(
                f"| {r.iteration} | `{h}` | {status}{current_marker} "
                f"| {avg_str} | {delta_str} | {val_str} |"
            )
        lines.append("")

        return "\n".join(lines)

    def format_past_agents(self) -> str:
        """Format a single markdown file containing all candidate summaries."""
        lines: list[str] = ["# Past Agents", ""]
        if not self._records:
            lines.append("No past agents recorded.")
            lines.append("")
            return "\n".join(lines)

        lines.append(
            "All candidate summaries with status, scores, evaluation tables, and agent code."
        )
        lines.append("")

        for i, record in enumerate(self._records):
            lines.append(self._format_candidate_overview(record).rstrip())
            if i != len(self._records) - 1:
                lines.append("")
                lines.append("---")
                lines.append("")

        if lines[-1] != "":
            lines.append("")
        return "\n".join(lines)

    # ---- Per-iteration overview ----

    def write_iteration_overview(self, iter_dir: str, iteration: int) -> None:
        """Write results.md using the same content as overview.md."""
        record = None
        for r in reversed(self._records):
            if r.iteration == iteration:
                record = r
                break
        if record is None:
            return

        overview_path = os.path.join(iter_dir, "results.md")
        os.makedirs(iter_dir, exist_ok=True)
        with open(overview_path, "w") as f:
            f.write(self._format_candidate_overview(record))
