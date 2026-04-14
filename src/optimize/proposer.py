import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import yaml
from openhands.sdk import Agent, Conversation, LLM, Tool
from openhands.sdk.context import Skill
from openhands.sdk.context.agent_context import AgentContext
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

from .common import (
    LiteralBlockDumper,
    add_to_cost_bucket,
    build_code_diff,
    hash_text,
    resolve_markdown_path,
    summarize_score_changes,
    validate_agent_candidate,
)
from ..utils import LM_DICT, build_sdk_llm
from ..task_setups import get_mcp_config
from ..runner import get_workspace_context

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

PROPOSER_SYSTEM_PROMPT = """You are an agent optimization expert. You improve an AI agent's task performance by analyzing execution trajectories and modifying the agent's configuration code.

## Workspace Layout

- `project/agent.py` — The current agent configuration code.
- `trajectories/` — Markdown traces of the agent's recent execution on training examples.
- `eval_results.yaml` — Per-example scores and detailed evaluation feedback.
- `proposal_memory/rejected_proposals.yaml` — Recent rejected edits for this candidate.
- `docs/sdk_reference.md` — High-level SDK map with numbered sections.
- `docs/sdk_reference_details/` — Per-section SDK signatures, examples, and implementation details.
{adaptation_guide_layout}

## Your Task

Diagnose why the agent is underperforming, then fix `project/agent.py` to improve it. Consult `docs/sdk_reference.md` to navigate the API surface, then read the relevant files under `docs/sdk_reference_details/` for concrete signatures and examples. Check `proposal_memory/rejected_proposals.yaml` before editing to avoid repeating non-improving changes.

## Constraints on agent.py

- MUST define `build_agent(base_dir: str, llm: LLM) -> Agent`.
    - `base_dir` is a temp directory where the function can write files (prompts, skills, etc.).
    - `llm` is the language model instance to use.
- Code must be valid, self-contained Python with explicit imports at the top.
- Use the staged SDK docs in `docs/` for the full API surface.
- After modifying agent.py, verify there are no syntax errors. Make sure the code is complete and self-contained.

{adaptation_guide_line}
"""

def build_proposer_system_prompt(use_adaptation_guide: bool) -> str:
    adaptation_guide_layout = ""
    adaptation_guide_line = ""
    if use_adaptation_guide:
        adaptation_guide_layout = "- `docs/adaptation.md` — Adaptation selection guide that maps failure modes to SDK sections and files."
        adaptation_guide_line = (
            "- If you need strategy guidance, use `docs/adaptation.md` to map failure modes "
            "to the relevant SDK sections and detail files."
        )
    return PROPOSER_SYSTEM_PROMPT.format(
        adaptation_guide_layout=adaptation_guide_layout,
        adaptation_guide_line=adaptation_guide_line,
    )

def _resolve_reference_doc_paths(
    use_adaptation_guide: bool = True,
    adaptation_guide_markdown: Optional[str] = None,
) -> dict[str, Path]:
    paths = {
        "docs_dir": DOCS_DIR,
        "sdk_reference": DOCS_DIR / "sdk_reference.md",
        "sdk_reference_details": DOCS_DIR / "sdk_reference_details",
    }
    if use_adaptation_guide:
        adaptation_path = DOCS_DIR / "adaptation.md"
        if adaptation_guide_markdown:
            adaptation_path = Path(resolve_markdown_path(adaptation_guide_markdown, str(DOCS_DIR)))
        paths["adaptation"] = adaptation_path
    return paths


def stage_proposer_reference_docs(
    workspace: str,
    use_adaptation_guide: bool = True,
    adaptation_guide_markdown: Optional[str] = None,
) -> None:
    source_paths = _resolve_reference_doc_paths(
        use_adaptation_guide=use_adaptation_guide,
        adaptation_guide_markdown=adaptation_guide_markdown,
    )
    docs_dest = Path(workspace) / "docs"
    docs_dest.mkdir(parents=True, exist_ok=True)

    sdk_reference = source_paths["sdk_reference"]
    if sdk_reference.exists():
        shutil.copy2(sdk_reference, docs_dest / "sdk_reference.md")

    sdk_reference_details = source_paths["sdk_reference_details"]
    if sdk_reference_details.exists():
        shutil.copytree(
            sdk_reference_details,
            docs_dest / "sdk_reference_details",
            dirs_exist_ok=True,
        )

    adaptation_path = source_paths.get("adaptation")
    if adaptation_path is not None and adaptation_path.exists():
        shutil.copy2(adaptation_path, docs_dest / "adaptation.md")


def _create_skill_file(content: str, name: str, dest_dir: str) -> Skill:
    skill_dir = os.path.join(dest_dir, name)
    os.makedirs(skill_dir, exist_ok=True)
    skill_file = os.path.join(skill_dir, "SKILL.md")
    with open(skill_file, "w") as f:
        f.write(f"---\nname: {name}\n---\n\n{content}")
    skill = Skill.load(path=skill_file, strict=False)
    skill.is_agentskills_format = False
    return skill


def load_proposer_skills(
    run_dir: str,
    use_adaptation_guide: bool = True,
    adaptation_guide_markdown: Optional[str] = None,
) -> list[Skill]:
    """Load proposer skills, optionally using a custom adaptation guide markdown file."""
    skills_dir = os.path.join(run_dir, "shared", "proposer_skills")
    os.makedirs(skills_dir, exist_ok=True)

    skills: list[Skill] = []
    source_paths = _resolve_reference_doc_paths(
        use_adaptation_guide=use_adaptation_guide,
        adaptation_guide_markdown=adaptation_guide_markdown,
    )

    sdk_ref_path = source_paths["sdk_reference"]
    if sdk_ref_path.exists():
        with open(sdk_ref_path) as f:
            sdk_content = f.read()
        skills.append(_create_skill_file(sdk_content, "sdk_reference", skills_dir))

    if use_adaptation_guide:
        adaptation_path = source_paths["adaptation"]
        if adaptation_path.exists():
            with open(adaptation_path) as f:
                adaptation_content = f.read()
            skills.append(_create_skill_file(adaptation_content, "adaptation_guide", skills_dir))

    return skills


class AgentProposer:
    def __init__(
        self,
        run_dir: str,
        task_prompt: str,
        model_name: str,
        reflection_lm_name: str,
        logger,
        cost_tracker: dict[str, dict],
        save_cost_summary: Callable[[], None],
        max_proposal_retries: int = 2,
        use_adaptation_guide: bool = True,
        adaptation_guide_markdown: Optional[str] = None,
        task_id: Optional[str] = None,
        use_docker: bool = False,
        server_image: str = "migration-analysis:latest",
        docker_network: Optional[str] = None,
    ):
        self.run_dir = run_dir
        self.task_prompt = task_prompt
        self.model_name = model_name
        self.reflection_lm_name = reflection_lm_name
        self.logger = logger
        self.cost_tracker = cost_tracker
        self.save_cost_summary = save_cost_summary
        self.max_proposal_retries = max_proposal_retries
        self.use_adaptation_guide = use_adaptation_guide
        self.adaptation_guide_markdown = adaptation_guide_markdown
        self.task_id = task_id
        self.use_docker = use_docker
        self.server_image = server_image
        self.docker_network = docker_network
        self._proposer_skills = load_proposer_skills(
            run_dir,
            use_adaptation_guide=use_adaptation_guide,
            adaptation_guide_markdown=adaptation_guide_markdown,
        )
        self._last_reflection_context: Optional[dict[str, Any]] = None
        self._pending_proposal_context: Optional[dict[str, Any]] = None

    def _proposal_memory_dir(self) -> str:
        return os.path.join(self.run_dir, "shared", "proposal_memory")

    def _proposal_memory_path(self) -> str:
        return os.path.join(self._proposal_memory_dir(), "rejected_proposals.jsonl")

    def _iter_proposal_metadata_path(self, iter_dir: str) -> str:
        return os.path.join(iter_dir, "proposer", "proposal_metadata.json")

    def _iter_proposal_outcome_path(self, iter_dir: str) -> str:
        return os.path.join(iter_dir, "proposer", "proposal_outcome.json")

    def _load_rejected_proposals(self, iteration: int) -> list[dict[str, Any]]:
        path = self._proposal_memory_path()
        if not os.path.exists(path):
            return []

        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    self.logger.log(f"[iter {iteration}] Skipping invalid proposal memory record")
        return records

    def _save_rejected_proposals(self, records: list[dict[str, Any]]):
        path = self._proposal_memory_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for record in records[-200:]:
                f.write(json.dumps(record, default=str))
                f.write("\n")

    def _append_rejected_proposal(self, iteration: int, record: dict[str, Any]):
        records = self._load_rejected_proposals(iteration)
        records.append(record)
        self._save_rejected_proposals(records)

    def _select_relevant_rejections(
        self,
        iteration: int,
        candidate: dict[str, str],
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        candidate_hash = hash_text(candidate["agent_code"])
        records = self._load_rejected_proposals(iteration)
        if not records:
            return []

        same_parent = [r for r in records if r.get("parent_candidate_hash") == candidate_hash]
        recent_other = [r for r in records if r.get("parent_candidate_hash") != candidate_hash]

        selected = same_parent[-limit:]
        remaining = max(0, limit - len(selected))
        if remaining > 0:
            selected.extend(recent_other[-min(2, remaining):])
        return selected

    def _write_proposal_metadata(self, iter_dir: str, record: dict[str, Any]):
        path = self._iter_proposal_metadata_path(iter_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def _write_proposal_outcome(self, iter_dir: str, record: dict[str, Any]):
        path = self._iter_proposal_outcome_path(iter_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def record_reflection_context(
        self,
        iteration: int,
        batch: Sequence[dict[str, Any]],
        candidate: dict[str, str],
        scores: Sequence[float],
    ) -> None:
        self._pending_proposal_context = None
        self._last_reflection_context = {
            "iteration": iteration,
            "parent_candidate_hash": hash_text(candidate["agent_code"]),
            "before_scores": list(scores),
            "before_score_sum": sum(scores),
            "batch_prompts": [example.get("prompt", "")[:300] for example in batch],
        }

    def record_candidate_outcome(
        self,
        iteration: int,
        iter_dir: str,
        batch: Sequence[dict[str, Any]],
        candidate: dict[str, str],
        scores: Sequence[float],
        outputs: Sequence[dict[str, Any]],
    ) -> None:
        pending = self._pending_proposal_context
        reflection = self._last_reflection_context
        if pending is None or reflection is None:
            return

        proposed_hash = hash_text(candidate["agent_code"])
        if proposed_hash != pending.get("proposed_candidate_hash"):
            return

        before_scores = list(reflection.get("before_scores", []))
        if len(before_scores) != len(scores):
            self.logger.log(
                f"[iter {iteration}] Skipping proposal outcome record due to score length mismatch"
            )
            self._pending_proposal_context = None
            return

        before_sum = float(reflection.get("before_score_sum", 0.0))
        after_score_list = list(scores)
        after_sum = sum(after_score_list)
        delta = after_sum - before_sum

        per_example_feedback = [
            {
                "prompt": example.get("prompt", "")[:300],
                "score": score,
                "feedback": output.get("feedback", "") if isinstance(output, dict) else "",
            }
            for example, score, output in zip(batch, scores, outputs)
        ]

        outcome = {
            **pending,
            "status": "accepted_on_subsample" if delta > 0 else "rejected_on_subsample",
            "batch_prompts": list(reflection.get("batch_prompts", [])),
            "before_scores": before_scores,
            "after_scores": after_score_list,
            "before_score_sum": before_sum,
            "after_score_sum": after_sum,
            "score_delta": delta,
            "score_delta_examples": summarize_score_changes(batch, before_scores, after_score_list),
            "per_example_feedback": per_example_feedback,
        }
        self._write_proposal_outcome(iter_dir, outcome)

        if delta <= 0:
            self._append_rejected_proposal(iteration, outcome)

        self._pending_proposal_context = None

    def _setup_workspace(
        self,
        iteration: int,
        iter_dir: str,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> str:
        workspace = os.path.join(iter_dir, "proposer", "workspace")
        if os.path.exists(workspace):
            shutil.rmtree(workspace)
        os.makedirs(workspace, exist_ok=True)
        os.chmod(workspace, 0o777)

        project_dir = os.path.join(workspace, "project")
        os.makedirs(project_dir)
        with open(os.path.join(project_dir, "agent.py"), "w") as f:
            f.write(candidate["agent_code"])

        traj_dir = os.path.join(workspace, "trajectories")
        os.makedirs(traj_dir)
        dataset = reflective_dataset.get("agent_code", [])
        for i, record in enumerate(dataset):
            md_parts = [f"# Example {i}"]
            md_parts.append(f"\n## Task Input\n{record.get('Task Input', '')}")
            md_parts.append(f"\n## Evaluation Feedback\n{record.get('Evaluation Feedback', '')}")
            trajectory_md = record.get("Agent Trajectory", "")
            if trajectory_md:
                md_parts.append(f"\n## Agent Trajectory\n{trajectory_md}")
            with open(os.path.join(traj_dir, f"example_{i:02d}.md"), "w") as f:
                f.write("\n".join(md_parts))

        eval_summary = []
        for i, record in enumerate(dataset):
            eval_out = record.get("eval_output", {})
            entry = {
                "example": i,
                "feedback": record.get("Evaluation Feedback", ""),
            }
            for key in ("task_id", "score", "answer", "reference_answers", "eval_types"):
                if key in eval_out:
                    entry[key] = eval_out[key]
            eval_summary.append(entry)
        with open(os.path.join(workspace, "eval_results.yaml"), "w") as f:
            yaml.dump(eval_summary, f, Dumper=LiteralBlockDumper, sort_keys=False, default_flow_style=False)

        memory_dir = os.path.join(workspace, "proposal_memory")
        os.makedirs(memory_dir)
        rejected_proposals = self._select_relevant_rejections(iteration, candidate)
        memory_payload = rejected_proposals or [
            {
                "status": "no_prior_rejections",
                "note": "No prior rejected proposals recorded for this candidate yet.",
            }
        ]
        with open(os.path.join(memory_dir, "rejected_proposals.yaml"), "w") as f:
            yaml.dump(memory_payload, f, Dumper=LiteralBlockDumper, sort_keys=False, default_flow_style=False)

        stage_proposer_reference_docs(
            workspace,
            use_adaptation_guide=self.use_adaptation_guide,
            adaptation_guide_markdown=self.adaptation_guide_markdown,
        )

        return workspace

    def _build_agent(self, workspace: str) -> Agent:
        lm = LM_DICT[self.reflection_lm_name]
        proposer_llm = build_sdk_llm(lm)
        proposer_tools = [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ]
        agent_context = AgentContext(skills=self._proposer_skills)

        # workspace is the docker path (e.g., /workspace/proposer) in docker mode
        # The prompt file should be relative to the workspace
        prompt_path = os.path.join(workspace, "system_prompt.md")

        mcp_cfg = get_mcp_config(self.task_id, workspace) if self.task_id else {}
        agent_kwargs = dict(
            llm=proposer_llm,
            tools=proposer_tools,
            system_prompt_filename=prompt_path,
            agent_context=agent_context,
        )
        if mcp_cfg:
            agent_kwargs["mcp_config"] = mcp_cfg
        return Agent(**agent_kwargs)

    def propose(
        self,
        iteration: int,
        iter_dir: str,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        new_candidate = dict(candidate)
        self._pending_proposal_context = None

        if "agent_code" not in components_to_update:
            return new_candidate

        workspace = self._setup_workspace(iteration, iter_dir, candidate, reflective_dataset)
        self.logger.log(f"[iter {iteration}] Proposer workspace: {workspace}")

        # Write prompt file to the host workspace directory
        prompt_path = os.path.join(workspace, "system_prompt.md")
        with open(prompt_path, "w") as f:
            f.write(build_proposer_system_prompt(self.use_adaptation_guide))

        with get_workspace_context(
            workspace_path=workspace,
            use_docker=self.use_docker,
            server_image=self.server_image,
            docker_network=self.docker_network,
            docker_workspace_path="/workspace/proposer",
            skills=self._proposer_skills,
        ) as (workspace_obj, workspace_path_for_agent):
            proposer_agent = self._build_agent(workspace_path_for_agent)
            instruction = "Diagnose why the agent is underperforming, then fix project/agent.py to improve it."

            conversation = Conversation(agent=proposer_agent, workspace=workspace_obj)
            conversation.send_message(instruction)
            self._run_proposer_loop(
                conversation, iteration, workspace, candidate, new_candidate, iter_dir, components_to_update
            )
            conversation.close()

        return new_candidate

    def _run_proposer_loop(
        self,
        conversation: Conversation,
        iteration: int,
        workspace: str,
        candidate: dict[str, str],
        new_candidate: dict[str, str],
        iter_dir: str,
        components_to_update: list[str],
    ) -> None:
        last_error = None
        for attempt in range(1 + self.max_proposal_retries):
            if attempt > 0:
                conversation.send_message(
                    "Your modified agent.py failed validation with this error:\n"
                    f"```\n{last_error}\n```\n"
                    "Please fix the code in project/agent.py and ensure "
                    "`build_agent(base_dir, llm)` returns a valid Agent."
                )

            try:
                conversation.run()
            except Exception as e:
                self.logger.log(f"[iter {iteration}] Proposer conversation error: {e}")
                traceback.print_exc()
                break

            agent_py_path = os.path.join(workspace, "project", "agent.py")
            try:
                with open(agent_py_path) as f:
                    new_code = f.read()
            except FileNotFoundError:
                self.logger.log(f"[iter {iteration}] Proposer did not produce agent.py")
                break

            success, error = validate_agent_candidate(new_code, self.model_name)
            if success:
                new_candidate["agent_code"] = new_code
                proposal_record = {
                    "iteration": iteration,
                    "parent_candidate_hash": hash_text(candidate["agent_code"]),
                    "proposed_candidate_hash": hash_text(new_code),
                    "components_to_update": list(components_to_update),
                    "code_diff": build_code_diff(candidate["agent_code"], new_code),
                }
                self._pending_proposal_context = proposal_record
                self._write_proposal_metadata(iter_dir, proposal_record)
                if attempt > 0:
                    self.logger.log(f"[iter {iteration}] Proposal succeeded on attempt {attempt + 1}")
                last_error = None
                break

            last_error = error
            self.logger.log(
                f"[iter {iteration}] Proposal attempt {attempt + 1} validation failed: {last_error}"
            )

        if last_error:
            self.logger.log(
                f"[iter {iteration}] All {1 + self.max_proposal_retries} proposal "
                "attempts failed, keeping current candidate"
            )

        try:
            proposer_log_dir = os.path.join(iter_dir, "proposer", "workspace_logs")
            os.makedirs(proposer_log_dir, exist_ok=True)
            from ..runner import save_conversation_trace

            save_conversation_trace(conversation, proposer_log_dir)
        except Exception as e:
            self.logger.log(f"[iter {iteration}] Failed to save proposer trace: {e}")

        try:
            proposer_metrics = conversation.conversation_stats.get_combined_metrics().get()
            add_to_cost_bucket(self.cost_tracker["proposer"], proposer_metrics)
            self.save_cost_summary()
        except Exception as e:
            self.logger.log(f"[iter {iteration}] Failed to collect proposer cost: {e}")
