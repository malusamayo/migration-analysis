import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from openhands.sdk import Agent, Conversation, LLM, Tool
from openhands.sdk.context import Skill
from openhands.sdk.context.agent_context import AgentContext
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

from .common import (
    add_to_cost_bucket,
    build_code_diff,
    hash_text,
    resolve_markdown_path,
    validate_agent_candidate,
)
from .memory import ProposerMemory
from ..utils import LM_DICT, build_sdk_llm
from ..task_setups import get_mcp_config, setup_proposer_workspace
from ..runner import get_workspace_context

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"

PROPOSER_SYSTEM_PROMPT = """You are an agent optimization expert. You improve an AI agent's task performance by analyzing execution trajectories and modifying the agent's configuration code.

## Workspace Layout

- `project/agent.py` — The current agent configuration code.
- `memory/scoreboard.md` — Succinct optimization history table: scores and status per candidate.
- `memory/current/overview.md` — Current candidate's code and evaluation results.
- `memory/current/trajectories/` — Raw JSON trajectories of the agent's recent execution on training examples. **Each file is a JSON array (list), not a dict** — use `events = json.load(open(...))` directly; do NOT do `json.load(...)['events']`. Use the `read_trajectory` skill for the full schema and ready-to-use Python snippets.
- `memory/past_agents.md` — Past candidate summaries with scores, evaluation results, and agent code.
- `docs/sdk_reference.md` — High-level SDK map with numbered sections.
- `docs/sdk_reference_details/` — Per-section SDK signatures, examples, and implementation details.
{adaptation_guide_layout}

## Constraints on agent.py

- MUST define `build_agent(base_dir: str, llm: LLM) -> Agent`.
    - `base_dir` is a temp directory where the function can write files (prompts, skills, etc.).
    - `llm` is the language model instance to use.
    - `system_prompt_filename` must be specified when constructing the Agent.
- Code must be valid, self-contained Python with explicit imports at the top.

## Additional Guidelines

- Past proposals and their outcomes are in `memory/past_agents.md`. Learn from them but do not feel constrained by them.
- Use the staged SDK docs in `docs/` for the full API surface if uncertain.
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
        "read_trajectory": DOCS_DIR / "read_trajectory.md",
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

    read_trajectory_path = source_paths["read_trajectory"]
    if read_trajectory_path.exists():
        shutil.copy2(read_trajectory_path, docs_dest / "read_trajectory.md")


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

    read_trajectory_path = source_paths["read_trajectory"]
    if read_trajectory_path.exists():
        with open(read_trajectory_path) as f:
            read_trajectory_content = f.read()
        skills.append(_create_skill_file(read_trajectory_content, "read_trajectory", skills_dir))

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
        memory: ProposerMemory,
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
        self.memory = memory
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

    def _iter_proposal_metadata_path(self, iter_dir: str) -> str:
        return os.path.join(iter_dir, "proposer", "proposal_metadata.json")

    def _write_proposal_metadata(self, iter_dir: str, record: dict[str, Any]):
        path = self._iter_proposal_metadata_path(iter_dir)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)

    def _setup_workspace(
        self,
        iteration: int,
        iter_dir: str,
        candidate: dict[str, str],
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

        self.memory.stage_workspace(workspace)

        stage_proposer_reference_docs(
            workspace,
            use_adaptation_guide=self.use_adaptation_guide,
            adaptation_guide_markdown=self.adaptation_guide_markdown,
        )

        if self.task_id:
            setup_proposer_workspace(self.task_id, workspace)

        # Make entire workspace world-writable so Docker's appuser can modify files
        for root, dirs, files in os.walk(workspace):
            os.chmod(root, 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o666)

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
        components_to_update: list[str],
        num_strategies: int = 1,
    ) -> list[dict[str, str]]:
        if "agent_code" not in components_to_update:
            return [dict(candidate)]

        workspace = self._setup_workspace(iteration, iter_dir, candidate)
        suffix = f" (x{num_strategies})" if num_strategies > 1 else ""
        self.logger.log(f"[iter {iteration}] Proposer workspace{suffix}: {workspace}")

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
            if num_strategies == 1:
                instruction = "Diagnose why the agent is underperforming, then fix project/agent.py to improve it."
            else:
                instruction = (
                    f"Diagnose why the agent is underperforming, then produce {num_strategies} diverse "
                    f"improvement strategies. Write strategy i to project/agent_i.py "
                    f"(project/agent_1.py, ..., project/agent_{num_strategies}.py). "
                    "Each strategy must explore a substantially different approach or area of improvement. "
                    "project/agent.py is the current agent for reference."
                )

            conversation = Conversation(agent=proposer_agent, workspace=workspace_obj)
            conversation.send_message(instruction)
            candidates = self._run_proposer_loop(
                conversation, iteration, workspace, candidate, iter_dir,
                components_to_update, num_strategies,
            )
            conversation.close()

        return candidates

    def _run_proposer_loop(
        self,
        conversation: Conversation,
        iteration: int,
        workspace: str,
        candidate: dict[str, str],
        iter_dir: str,
        components_to_update: list[str],
        num_strategies: int = 1,
    ) -> list[dict[str, str]]:
        agent_llm = build_sdk_llm(LM_DICT[self.model_name])
        valid_candidates: list[dict[str, str]] = []
        last_error = None

        for attempt in range(1 + self.max_proposal_retries):
            if attempt > 0:
                if num_strategies == 1:
                    conversation.send_message(
                        "Your modified agent.py failed validation with this error:\n"
                        f"```\n{last_error}\n```\n"
                        "Please fix the code in project/agent.py and ensure "
                        "`build_agent(base_dir, llm)` returns a valid Agent."
                    )
                else:
                    conversation.send_message(
                        "None of the strategy files passed validation. "
                        "Please fix them and ensure each `build_agent(base_dir, llm)` returns a valid Agent."
                    )

            try:
                conversation.run()
            except Exception as e:
                self.logger.log(f"[iter {iteration}] Proposer conversation error: {e}")
                traceback.print_exc()
                break

            if num_strategies == 1:
                try:
                    with open(os.path.join(workspace, "project", "agent.py")) as f:
                        new_code = f.read()
                    files_to_validate = [(None, new_code)]
                except FileNotFoundError:
                    self.logger.log(f"[iter {iteration}] Proposer did not produce agent.py")
                    break
            else:
                files_to_validate = []
                for k in range(1, num_strategies + 1):
                    path = os.path.join(workspace, "project", f"agent_{k}.py")
                    if os.path.exists(path):
                        with open(path) as f:
                            files_to_validate.append((k, f.read()))
                    else:
                        self.logger.log(f"[iter {iteration}] Strategy {k} file not found, skipping")

            valid_candidates = []
            for idx, new_code in files_to_validate:
                success, error = validate_agent_candidate(new_code, agent_llm)
                if success:
                    new_cand = dict(candidate)
                    new_cand["agent_code"] = new_code
                    code_diff = build_code_diff(candidate["agent_code"], new_code)
                    self.memory.record_proposal(
                        iteration=iteration,
                        parent_code=candidate["agent_code"],
                        proposed_code=new_code,
                        components_updated=list(components_to_update),
                        code_diff=code_diff,
                    )
                    proposal_record = {
                        "iteration": iteration,
                        "parent_candidate_hash": hash_text(candidate["agent_code"]),
                        "proposed_candidate_hash": hash_text(new_code),
                        "components_to_update": list(components_to_update),
                        "code_diff": code_diff,
                    }
                    if idx is not None:
                        proposal_record["strategy_index"] = idx
                        metadata_dir = os.path.join(iter_dir, f"strategy_{idx}")
                    else:
                        metadata_dir = iter_dir
                    self._write_proposal_metadata(metadata_dir, proposal_record)
                    if attempt > 0 and num_strategies == 1:
                        self.logger.log(f"[iter {iteration}] Proposal succeeded on attempt {attempt + 1}")
                    valid_candidates.append(new_cand)
                else:
                    if num_strategies == 1:
                        last_error = error
                    label = f"Strategy {idx}" if idx is not None else f"Attempt {attempt + 1}"
                    self.logger.log(f"[iter {iteration}] {label} validation failed: {error}")

            if valid_candidates:
                if num_strategies > 1:
                    self.logger.log(
                        f"[iter {iteration}] {len(valid_candidates)}/{num_strategies} strategies valid"
                    )
                break

            if num_strategies > 1:
                self.logger.log(f"[iter {iteration}] Attempt {attempt + 1}: no valid strategies, retrying")

        if not valid_candidates:
            self.logger.log(
                f"[iter {iteration}] All {1 + self.max_proposal_retries} proposal "
                "attempts failed, keeping current candidate"
            )
            valid_candidates = [dict(candidate)]

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

        return valid_candidates
