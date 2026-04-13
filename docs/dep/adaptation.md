# Meta-Agent Adaptation Selection Guide

When configuring an agent for a task, analyze the task and model capabilities, then select adaptations from the following catalog. Each adaptation addresses a specific gap between what the task requires and what the model can do out of the box.

Adaptations are organized into five high-level themes:

1. [Augment Context](#1-augment-context) — Give the agent more knowledge
2. [Manage Scope and Context](#2-manage-scope-and-context) — Reduce or restructure what the agent must attend to
3. [Adapt Actions](#3-adapt-actions) — Change what the agent can do
4. [Adapt Observations](#4-adapt-observations) — Change what the agent sees back
5. [Adapt Environment or Orchestration Loop](#5-adapt-environment-or-orchestration-loop) — Change how the agent is run

---

## 1. Augment Context

Add information to the agent's context so it makes better decisions.

### 1a. Add Instructions

**When to use:** The model selects incorrect actions or follows a wrong process because it lacks explicit procedural guidance or domain knowledge.

**SDK implementation:** `AgentContext.system_message_suffix`, `AgentContext.user_message_suffix`, `Skill(trigger=None)` (always-active repo skills), or a custom `system_prompt_filename` on `Agent`. See SDK Reference §5 (AgentContext) and §6 (Skill).

**Task-side signals:**
- The task has clear rules or heuristics for which action to take in which situation
- The task has implicit domain knowledge that can be made explicit (e.g., conventions, constraints, verification steps)
- The task benefits from explicit self-validation or process-adherence steps

**Trajectory-side signals:**
- The model chooses the wrong action despite having the right tools available
- The model skips verification, omits checks, or does not self-correct
- Errors stem from the model not knowing *what to do*, not from tool failures

### 1b. Add Examples

**When to use:** The model has the right tools but fails to pick the correct action given the context, and the mapping from context to action is best demonstrated rather than described.

**SDK implementation:** `Skill` content containing few-shot examples (with `trigger=None` for always-on, or `KeywordTrigger`/`TaskTrigger` for conditional injection), or `Agent.system_prompt_kwargs` passed to a custom Jinja2 template. See SDK Reference §6 (Skill).

**Task-side signals:**
- The task relies on implicit domain knowledge that is easier to show than tell
- Action sequences follow reusable patterns the model should reproduce
- Tool semantics are complex enough that usage is best conveyed by example

**Trajectory-side signals:**
- The model makes unpredictable or unreliable action choices
- Errors stem from the model choosing the wrong action, not from the action itself failing
- The model misuses tools (wrong parameters, wrong sequencing) despite correct schemas

---

## 2. Manage Scope and Context

Reduce or restructure the information the agent must process so it can focus on what matters.

### 2a. Retrieve Contexts When Needed / Progressive Disclosure

**When to use:** The agent does not need all context upfront — relevant information can be fetched on demand or revealed progressively as the task narrows, keeping the working context small.

**SDK implementation:** `Skill` with `KeywordTrigger` or `TaskTrigger` (content injected only when relevant keywords/task types appear). See SDK Reference §6 (Skill).

**Task-side signals:**
- The task has a large information space but only a small subset is relevant at any given step
- Relevant context depends on earlier decisions and cannot be determined in advance
- The task has a funneling structure where broad exploration narrows into focused execution

**Trajectory-side signals:**
- The agent is overwhelmed by upfront context and fixates on irrelevant details
- The agent performs better when given less initial context and allowed to query for more
- Early steps involve information gathering that could replace pre-loaded context

### 2b. Delegate to Subagents

**When to use:** The task decomposes into subtasks that are sufficiently independent that each benefits from its own isolated context and tool set — either because they can run in parallel, or because cross-contamination of context between them causes errors.

> **Critical distinction from 2c:** Delegation is about *independence*. If the subtasks must share intermediate state or depend tightly on each other's outputs during execution, delegation adds coordination overhead without payoff — use 2c (explicit planning phase) or the default loop instead.

**SDK implementation:** Agent Delegation via `DelegateTool` and `register_agent`. Each sub-agent gets its own `Agent` instance with independent tools and context. See SDK Reference §12 (Agent Delegation).

**Task-side signals:**
- The task has clear, separable subtasks with minimal shared state between them
- Subtasks benefit from independent trial-and-error or self-validation
- Different subtasks require different tool sets or expertise

**Trajectory-side signals:**
- The main agent is overwhelmed by context length or too many tool options
- The agent conflates unrelated subtasks or loses track of its current objective
- Performance degrades as the trajectory grows longer

### 2c. Explicit Planning Phase

**When to use:** The task benefits from a dedicated planning phase — producing a structured plan before any execution begins — rather than interleaving planning and execution (the default ReAct loop). Use this when the full plan can be determined without executing actions first, and when having a reviewable plan upfront improves downstream execution quality.

> **Critical distinction from 2b:** This is not about multiple agents — it is about multiple phases within a single logical task. A planner produces a plan; an executor follows it. No parallel execution, no context isolation.
> **Critical distinction from default loop:** The default agent loop interleaves reasoning and action (ReAct style). The explicit planning phase deliberately separates them: the planner does not execute, the executor does not re-plan from scratch. If execution reveals information that would invalidate the plan, the default ReAct loop is a better fit.

**SDK implementation:** Two-Phase Workflow pattern using separate `Conversation` instances (e.g., `get_planning_agent` then `get_default_agent`), or `TaskTrackerTool` for in-agent phase tracking. See SDK Reference §16 (Common Patterns: Two-Phase Workflow) and §10 (Built-in Tools: TaskTrackerTool).

**Task-side signals:**
- The task has a natural plan-then-execute structure where the plan can be fully specified before any action is taken
- The plan is valuable as a human-reviewable artifact (approval before execution begins)
- The full task exceeds the agent's effective planning horizon when approached reactively

**Trajectory-side signals:**
- The agent dives into execution prematurely and must frequently backtrack or restart
- The agent conflates planning ("what should I do") with execution ("doing it"), leading to incoherent sequences
- When manually given a step-by-step plan before execution, the agent succeeds; without it, it fails

### 2d. Active Context Compression

**When to use:** The agent runs many steps and accumulated context is degrading performance, but the task does not decompose cleanly into independent subtasks.

**SDK implementation:** `LLMSummarizingCondenser` (or `PipelineCondenser` chaining multiple condensers). Configure `max_size`, `keep_first`, and `max_tokens` to control when condensation triggers. See SDK Reference §8 (Condensers).

**Task-side signals:**
- The task requires a long sequence of actions (high step count)
- Earlier steps produce information that can be summarized without loss
- The task's critical information is sparse relative to total context

**Trajectory-side signals:**
- The agent starts making errors late in the trajectory despite early steps being correct
- Context window is filling up, causing truncation or degraded attention
- The agent re-reads or re-discovers information it already processed

---

## 3. Adapt Actions

Change the tools available to the agent — add, remove, or modify them.

### 3a. Create Tools

**When to use:** The task requires actions the model cannot currently perform, or existing actions are too token-expensive (too many steps or too verbose per step).

**SDK implementation:** Custom Tool Definition Pattern (Action → Observation → ToolExecutor → ToolDefinition → `register_tool`), or MCP Integration via `Agent.mcp_config` to add external tool servers. See SDK Reference §7 (Custom Tool Definition Pattern) and §11 (MCP Integration).

**Task-side signals:**
- The task involves repeated subroutine-like patterns that could be collapsed into a single call
- Required capabilities are missing entirely from the current tool set
- Actions are overly complex and could be simplified behind a tool interface

**Trajectory-side signals:**
- The agent attempts multi-step workarounds for what should be a single operation
- The agent fails because it simply cannot perform a required action
- Token budget is consumed by verbose, repetitive action sequences

### 3b. Remove Tools

**When to use:** The agent has access to too many tools, and irrelevant tools are degrading selection accuracy or tempting the agent into unproductive paths.

**SDK implementation:** `Agent.filter_tools_regex` to filter resolved tools by name regex, or `Agent.include_default_tools=[]` to remove built-in defaults. See SDK Reference §2 (Agent).

**Task-side signals:**
- The task only requires a small subset of available tools
- Some tools are dangerous or irrelevant for this task class
- Tools have overlapping functionality that creates ambiguity

**Trajectory-side signals:**
- The agent selects irrelevant tools when simpler or more appropriate ones exist
- High tool count is causing selection errors or decision paralysis
- The agent uses a tool-search mechanism but still gets distracted by irrelevant results

### 3c. Update Tool Schemas

**When to use:** The model invokes tools but does so incorrectly — wrong parameters, malformed calls, or selecting the wrong tool from a set — due to unclear or misleading schemas.

**SDK implementation:** Modify `ToolDefinition.description`, `Action` field descriptions/types, and `ToolAnnotations` (readOnlyHint, destructiveHint, etc.) to clarify tool semantics. See SDK Reference §7 (Custom Tool Definition Pattern) and §3 (Tool Spec).

**Task-side signals:**
- Tools have parameter constraints or semantics not captured in the current schema
- Multiple tools have overlapping signatures that create confusion
- Tool behavior depends on parameter combinations that are not self-evident

**Trajectory-side signals:**
- Tool call failures stem from schema misunderstanding, not from choosing the wrong strategy
- The model passes wrong parameter types, omits required fields, or confuses similar tools
- Adding usage instructions alone does not resolve the invocation errors

---

## 4. Adapt Observations

Change what the agent sees in tool outputs before it processes them.

### 4a. Trim / Filter Observations

**When to use:** Tool outputs are too large or too noisy, consuming context the agent needs for subsequent reasoning.

**SDK implementation:** Override `Observation.to_llm_content` on custom observation classes to filter, truncate, or selectively surface information before the LLM sees it. Also consider `LLM.max_message_chars` as a hard cap. See SDK Reference §7 (Custom Tool Definition Pattern, Step 2: Define Observation).

**Task-side signals:**
- Tool outputs are verbose relative to what the agent actually needs
- Outputs contain boilerplate, logs, or metadata that is irrelevant to the task
- The useful signal in outputs can be extracted by simple filtering rules

**Trajectory-side signals:**
- The agent runs many steps and risks running out of context
- The agent is distracted by irrelevant parts of tool output
- Removing verbose output sections would not change the agent's optimal next action

### 4b. Post-Process / Transform Observations

**When to use:** Tool outputs are in a format the model struggles with, or critical information needs to be extracted, restructured, or summarized before the agent can use it effectively.

**SDK implementation:** Override `Observation.to_llm_content` to reformat, summarize, paginate, or annotate outputs. Return `TextContent` and/or `ImageContent` objects tailored to the LLM's strengths. See SDK Reference §7 (Custom Tool Definition Pattern, Step 2: Define Observation).

**Task-side signals:**
- Outputs are in formats the model handles poorly (e.g., raw binary, very wide tables, deeply nested structures)
- Critical information is buried in large outputs and could be surfaced by summarization or extraction
- Outputs could be paginated, ranked, or annotated to guide the agent's attention

**Trajectory-side signals:**
- The agent misinterprets or ignores important information in tool output
- The agent repeatedly re-invokes tools to find information that was already returned
- Performance improves when outputs are manually reformatted before the agent sees them

---

## 5. Adapt Environment or Orchestration Loop

Change how the agent is run, supervised, or controlled.

### 5a. Add Monitoring

**When to use:** The task benefits from external oversight of the agent's behavior — tracking progress, detecting stalls, or logging actions for later review.

**SDK implementation:** `Conversation.callbacks` for per-event observation, `conversation.conversation_stats` for cost/token metrics, `stuck_detection=True` with `stuck_detection_thresholds` for automatic stall detection, and `Critic` (experimental) for real-time evaluation. See SDK Reference §14 (Events and Callbacks), §4 (Conversation), and §13 (Critic).

**Task-side signals:**
- The task has measurable progress indicators that can be checked externally
- The task has time or resource constraints that should trigger intervention
- Post-hoc analysis of agent behavior is valuable for future adaptation

**Trajectory-side signals:**
- The agent stalls or loops without making progress
- The agent consumes excessive resources (tokens, API calls, time) without detection
- Failures are only discovered after the full trajectory completes

### 5b. Action Intercept

**When to use:** The task involves actions that are potentially irreversible or harmful, and the agent may lack the domain knowledge to assess risk on its own.

**SDK implementation:** `HookConfig.pre_tool_use` hooks that inspect and allow/deny tool calls before execution, `conversation.set_confirmation_policy()` for human-in-the-loop approval, and `ToolAnnotations` (destructiveHint, readOnlyHint) to declare risk levels. See SDK Reference §9 (Hooks) and §7 (ToolAnnotations).

**Task-side signals:**
- Some actions have side effects that cannot be undone (deletions, deployments, sends)
- The task involves implicit safety constraints the model may not fully internalize
- The environment has guardrails that should be enforced externally rather than relying on the model

**Trajectory-side signals:**
- The agent takes risky actions without hesitation or verification
- The agent misunderstands the consequences of certain actions
- Safety-critical errors occur that would have been caught by a simple rule-based check

---

## Selection Process

When deciding which adaptations to apply:

1. **Identify failure modes.** Run the agent on representative tasks and observe where it fails — wrong actions, failed tool calls, missing capabilities, bloated context, unsafe actions, or broken processes.
2. **Map failures to adaptations.** Use the task-side and trajectory-side signals above to match each observed failure to one or more adaptations.
3. **Prefer lightweight fixes first.** Instructions and examples (1a, 1b) are cheaper to implement than new tools (3a) or subagents (2b). Start there unless the gap clearly requires structural changes.
4. **Combine as needed.** Adaptations are not mutually exclusive. A complex task may need new tools (3a), examples (1b), process instructions (1a), and observation trimming (4a) simultaneously.
5. **Re-evaluate after each change.** Each adaptation shifts the agent's behavior. Re-test to confirm the fix works and hasn't introduced new failure modes.

---

## Strategy Relationships

### Decision flow for scope and context failures

When the agent fails due to context or scope problems, use the following to choose between 2a–2d:

```
Are the subtasks sufficiently independent (parallel or isolated)?
├─ Yes → 2b (Delegate to Subagents)
└─ No (subtasks share state or are tightly coupled)
     ├─ Can the full plan be determined before any execution begins?
     │   ├─ Yes → 2c (Explicit Planning Phase)
     │   └─ No (planning requires interleaving with execution) → default ReAct loop; no structural change
     ├─ Is context growing too large due to accumulated trajectory history? → 2d (Active Context Compression)
     └─ Is the agent overwhelmed by too much upfront information? → 2a (Progressive Disclosure)
```

### Implementation cost ladder

Prefer cheaper adaptations first; escalate only when the cheaper fix is insufficient.

`1a < 1b < 3b < 3c < 2a < 4a < 4b < 2d < 2c < 3a < 2b < 5a < 5b`

### How adaptations combine

- **2b + 2c:** Use 2c (planning phase) to produce a task graph, then 2b to delegate independent nodes in that graph to subagents. Planning determines the decomposition; delegation executes it in parallel.
- **1a → 2c:** Adding explicit planning instructions (1a) is the lightweight version of 2c. If instructions alone don't produce structured enough plans, escalate to a full two-phase workflow.
- **4a/4b → 2d:** Both address bloated context at different points — 4a/4b prevent large observations from entering context; 2d compresses history after it has accumulated. Prefer 4a/4b first (prevention over cure).
- **2c + 5b:** When the plan is a reviewable artifact, action interception (5b) can be applied at plan-approval time rather than per-action — more efficient oversight for high-stakes tasks.

