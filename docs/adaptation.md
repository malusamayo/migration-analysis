# Adaptation Selection Guide

This guide helps a meta-agent diagnose SLM agent failures and select targeted adaptations.

All SDK references below use the numbering in [`sdk_reference.md`](sdk_reference.md). The main file is the high-level overview; signatures and examples live in `docs/sdk_reference_details/`.

---

## 1. Tool-Use Failures

**Observable SLM behaviors:**
- Schema violations: malformed JSON, missing required fields, wrong parameter types
- Tool hallucination: invokes non-existent tools by composing plausible-sounding names
- Wrong tool selection: picks a semantically adjacent but functionally incorrect tool
- Wrong tool configuration selection: selects the right tool but with wrong parameters
- Anchors on familiar API patterns and breaks on unfamiliar schemas

### Recommended adaptations

#### 1a. Create new tools

When the SLM struggles to use existing tools correctly, the most effective fix is often to create new tools with clearer semantics, simpler interface, and more high-level abstraction. This reduces the cognitive load on the SLM and guides it toward correct usage.

The created tools can:
- Enforce the correct schema through constrained parameters and clear descriptions
- Abstract away low-level details that the SLM struggles with, collapsing multi-step subroutines
- Standardize the interface to novel APIs, reducing hallucination and selection errors

**SDK references:** SDK Reference §7, Custom Tool Definition Pattern, in [`sdk_reference.md`](sdk_reference.md#7-custom-tool-definition-pattern) with implementation details in [`sdk_reference_details/07_custom_tool_definition_pattern.md`](sdk_reference_details/07_custom_tool_definition_pattern.md). For external tool servers, also see SDK Reference §11, MCP Integration, in [`sdk_reference.md`](sdk_reference.md#11-mcp-integration) with implementation details in [`sdk_reference_details/11_mcp_integration.md`](sdk_reference_details/11_mcp_integration.md).

#### 1b. Filter tools

Restrict the active tool set to only what the current task or step requires. SLMs degrade significantly beyond roughly 10 to 15 active tools.

**SDK references:** SDK Reference §2, Agent, in [`sdk_reference.md`](sdk_reference.md#2-agent) with implementation details in [`sdk_reference_details/02_agent.md`](sdk_reference_details/02_agent.md). The relevant fields are `filter_tools_regex` and `include_default_tools`.

#### 1c. Add tool-use examples

Provide few-shot demonstrations of correct tool invocations, especially for novel tools. Show the full format including parameter values.

**SDK references:** SDK Reference §5, AgentContext, in [`sdk_reference.md`](sdk_reference.md#5-agentcontext) with implementation details in [`sdk_reference_details/05_agent_context.md`](sdk_reference_details/05_agent_context.md), and SDK Reference §6, Skill, in [`sdk_reference.md`](sdk_reference.md#6-skill) with implementation details in [`sdk_reference_details/06_skill.md`](sdk_reference_details/06_skill.md).

---

## 2. Instruction-Following Failures

**Observable SLM behaviors:**
- Partial compliance: some constraints satisfied, others silently dropped
- Late-trajectory drift: instructions followed in early turns and abandoned later
- Output format violations: malformed JSON or XML, truncated responses, or missing required fields

### Recommended adaptations

#### 2a. Reinforce instructions statically

Make key constraints and format requirements explicit in the system prompt or as a fixed user message. Vary the phrasing and position of instructions to find what works best for the SLM.

**SDK references:** SDK Reference §2, Agent, in [`sdk_reference.md`](sdk_reference.md#2-agent) with implementation details in [`sdk_reference_details/02_agent.md`](sdk_reference_details/02_agent.md), plus SDK Reference §5, AgentContext, in [`sdk_reference.md`](sdk_reference.md#5-agentcontext) with implementation details in [`sdk_reference_details/05_agent_context.md`](sdk_reference_details/05_agent_context.md).

#### 2b. Reinforce instructions dynamically

Re-inject key constraints at regular intervals, for example after a specific tool call or every N steps. Keep reinforcements short and structured.

**SDK references:** SDK Reference §9, Hooks, in [`sdk_reference.md`](sdk_reference.md#9-hooks) with implementation details in [`sdk_reference_details/09_hooks.md`](sdk_reference_details/09_hooks.md).

#### 2c. Enforce instructions through tools

Encode constraints directly into tool schemas, output types, or workflow guards so they cannot be silently violated. For output format issues, use structured output templates.

**SDK references:** SDK Reference §7, Custom Tool Definition Pattern, in [`sdk_reference.md`](sdk_reference.md#7-custom-tool-definition-pattern) with implementation details in [`sdk_reference_details/07_custom_tool_definition_pattern.md`](sdk_reference_details/07_custom_tool_definition_pattern.md).

---

## 3. Implicit-Knowledge Failures

**Observable SLM behaviors:**
- Misses implied constraints or conventions
- Skips conventional steps that a domain practitioner would perform automatically
- Tries multiple incorrect approaches that a knowledgeable agent would know to avoid
- Fails to leverage domain-specific heuristics that would guide problem-solving

### Recommended adaptations

#### 3a. Externalize domain knowledge as instructions

Make domain conventions, environment assumptions, and unstated constraints explicit. Start with the most frequently violated conventions and add more as failures are observed.

**SDK references:** SDK Reference §5, AgentContext, in [`sdk_reference.md`](sdk_reference.md#5-agentcontext) with implementation details in [`sdk_reference_details/05_agent_context.md`](sdk_reference_details/05_agent_context.md), plus SDK Reference §2, Agent, in [`sdk_reference.md`](sdk_reference.md#2-agent) with implementation details in [`sdk_reference_details/02_agent.md`](sdk_reference_details/02_agent.md) for custom prompt templates.

#### 3b. Inject knowledge progressively via skills or progressive disclosure

Rather than front-loading all domain knowledge, reveal it as the task narrows. Inject context when specific keywords or task states are detected.

**SDK references:** SDK Reference §6, Skill, in [`sdk_reference.md`](sdk_reference.md#6-skill) with implementation details in [`sdk_reference_details/06_skill.md`](sdk_reference_details/06_skill.md).

#### 3c. Add task-matched examples

When domain knowledge is easier to show than tell, provide few-shot demonstrations that illustrate the implicit conventions in action.

**SDK references:** SDK Reference §5, AgentContext, in [`sdk_reference.md`](sdk_reference.md#5-agentcontext) with implementation details in [`sdk_reference_details/05_agent_context.md`](sdk_reference_details/05_agent_context.md), and SDK Reference §6, Skill, in [`sdk_reference.md`](sdk_reference.md#6-skill) with implementation details in [`sdk_reference_details/06_skill.md`](sdk_reference_details/06_skill.md).

---

## 4. Long-Context Failures

**Observable SLM behaviors:**
- Degrades on later steps of a long trajectory while early steps are fine
- Context anxiety: premature task termination as the context window fills, producing incomplete outputs
- Re-queries information already retrieved earlier

### Recommended adaptations

#### 4a. Retrieve context on demand

Fetch domain documentation, reference material, or past successful trajectories at each decision step rather than pre-loading everything. 

When simple regex-based retrieval is insufficient, consider implementing established retriever patterns such as BM-25 or dense retrieval with a vector database.

**SDK references:** SDK Reference §7, Custom Tool Definition Pattern, in [`sdk_reference.md`](sdk_reference.md#7-custom-tool-definition-pattern) with implementation details in [`sdk_reference_details/07_custom_tool_definition_pattern.md`](sdk_reference_details/07_custom_tool_definition_pattern.md).

#### 4b. Trim observations

Reduce tool output size before it enters the context. Extract only the fields relevant to the current step.

**SDK references:** SDK Reference §7, Custom Tool Definition Pattern, in [`sdk_reference.md`](sdk_reference.md#7-custom-tool-definition-pattern) with implementation details in [`sdk_reference_details/07_custom_tool_definition_pattern.md`](sdk_reference_details/07_custom_tool_definition_pattern.md). The key extension point is `Observation.to_llm_content`.

#### 4c. Compress accumulated context

Periodically summarize trajectory history to keep the effective context length manageable.

**SDK references:** SDK Reference §8, Condensers, in [`sdk_reference.md`](sdk_reference.md#8-condensers) with implementation details in [`sdk_reference_details/08_condensers.md`](sdk_reference_details/08_condensers.md).

#### 4d. Delegate to subagents

Assign subtasks to fresh SLM instances with clean context windows. Each subagent sees only the tools and context relevant to its subtask.

**SDK references:** SDK Reference §12, Agent Delegation, in [`sdk_reference.md`](sdk_reference.md#12-agent-delegation) with implementation details in [`sdk_reference_details/12_agent_delegation.md`](sdk_reference_details/12_agent_delegation.md).

---

## 5. Planning Failures

**Observable SLM behaviors:**
- Omits critical subtasks from the initial plan
- Commits to an incorrect plan without recovery
- Spins in replanning loops that make no forward progress

### Recommended adaptations

#### 5a. Create a plan statically with a frontier LLM

Use a frontier LLM to generate a concrete step-by-step plan; the SLM executes without needing to construct task structure. This is most effective when tasks are structurally repetitive and share the same plan skeleton.

**SDK references:** Treat meta-agent as a planner that produces a structured plan -- this can be injected as a fixed system prompt, a created script, or a new tool. The SLM worker then executes the plan with no additional reasoning burden. SDK Reference §2, Agent, in [`sdk_reference.md`](sdk_reference.md#2-agent) with implementation details in [`sdk_reference_details/02_agent.md`](sdk_reference_details/02_agent.md) for custom prompt templates or custom tools, plus SDK Reference §5, AgentContext, in [`sdk_reference.md`](sdk_reference.md#5-agentcontext) with implementation details in [`sdk_reference_details/05_agent_context.md`](sdk_reference_details/05_agent_context.md) for custom system prompts.

#### 5b. Plan-then-execute with a frontier LLM

If a dynamic plan is needed, use a frontier LLM to generate a full plan at the start of execution, then have the SLM execute step-by-step with the plan visible in context. This offloads the planning burden to the stronger model while keeping the SLM focused on execution.

**SDK references:** SDK Reference §16, Common Patterns, in [`sdk_reference.md`](sdk_reference.md#16-common-patterns) with implementation details in [`sdk_reference_details/16_common_patterns.md`](sdk_reference_details/16_common_patterns.md), plus SDK Reference §4, Conversation, in [`sdk_reference.md`](sdk_reference.md#4-conversation) with implementation details in [`sdk_reference_details/04_conversation.md`](sdk_reference_details/04_conversation.md) for the multi-stage execution loop.
