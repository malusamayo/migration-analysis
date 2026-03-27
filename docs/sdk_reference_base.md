# OpenHands Agent SDK Reference

This document is a practical API reference for writing code that constructs and runs agents using the OpenHands Agent SDK. All import paths, class signatures, and parameters are documented for direct use.

---

## Table of Contents

1. [LLM](#1-llm)
2. [Agent](#2-agent)
3. [Tool (Spec)](#3-tool-spec)
4. [Conversation](#4-conversation)
5. [AgentContext](#5-agentcontext)
6. [Skill](#6-skill)
7. [Custom Tool Definition Pattern](#7-custom-tool-definition-pattern)
8. [Condensers](#8-condensers)
9. [Hooks](#9-hooks)
10. [Built-in Tools](#10-built-in-tools)
11. [MCP Integration](#11-mcp-integration)
12. [Agent Delegation](#12-agent-delegation)
13. [Critic (Experimental)](#13-critic-experimental)
14. [Events and Callbacks](#14-events-and-callbacks)
15. [LLM Registry](#15-llm-registry)
16. [Common Patterns](#16-common-patterns)

---

## 1. LLM

**Import:** `from openhands.sdk import LLM` (or `from openhands.sdk.llm import LLM`)

The LLM class wraps litellm to provide a unified interface for interacting with language models.

### Constructor Signature

```python
class LLM(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    api_key: str | SecretStr | None = None
    base_url: str | None = None
    api_version: str | None = None          # For Azure
    usage_id: str = "default"               # Unique ID for registry/telemetry/spend tracking
    temperature: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_message_chars: int = 30_000         # Max chars per event/content sent to LLM
    num_retries: int = 5
    retry_multiplier: float = 8.0
    retry_min_wait: int = 8
    retry_max_wait: int = 64
    timeout: int | None = 300               # HTTP timeout in seconds
    stream: bool = False                    # Enable streaming responses
    caching_prompt: bool = True
    native_tool_calling: bool = True
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "none"] | None = "high"
    extended_thinking_budget: int | None = 200_000  # Anthropic thinking budget
    seed: int | None = None
    disable_vision: bool | None = None
    model_canonical_name: str | None = None  # For feature registry lookups with proxied models
    extra_headers: dict[str, str] | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    log_completions: bool = False
    custom_tokenizer: str | None = None
    litellm_extra_body: dict[str, Any] = {}  # Extra params for litellm (vLLM, proxy, etc.)
    fallback_strategy: FallbackStrategy | None = None  # Alternate LLMs on failure
    drop_params: bool = True
    # AWS Bedrock
    aws_access_key_id: str | SecretStr | None = None
    aws_secret_access_key: str | SecretStr | None = None
    aws_region_name: str | None = None
```

### Key Properties

- `llm.metrics` - Access `Metrics` object with `accumulated_cost`, token counts, etc.
- `llm.model_copy(update={...})` - Create a copy with modified fields (Pydantic)

### Direct Completion

```python
from openhands.sdk import LLM, Message, TextContent

llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key="...")
resp = llm.completion(
    messages=[Message(role="user", content=[TextContent(text="Hello")])]
)
# resp is LLMResponse; resp.message is Message
texts = [c.text for c in resp.message.content if isinstance(c, TextContent)]
```

### Example

```python
import os
from openhands.sdk import LLM

llm = LLM(
    model=os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", None),
)
```

---

## 2. Agent

**Import:** `from openhands.sdk import Agent` (or `from openhands.sdk.agent import Agent`)

The Agent class is the core orchestrator. It is stateless and configured with an LLM, tools, optional context, condenser, and critic.

### Constructor Signature

```python
class Agent(CriticMixin, AgentBase):
    llm: LLM                                              # Required
    tools: list[Tool] = []                                 # Tool specs to resolve at init
    mcp_config: dict[str, Any] = {}                        # MCP server configuration
    filter_tools_regex: str | None = None                  # Regex to filter resolved tools by name
    include_default_tools: list[str] = ["FinishTool", "ThinkTool"]  # Built-in tools to include
    agent_context: AgentContext | None = None               # Skills, secrets, prompt extensions
    system_prompt_filename: str = "system_prompt.j2"       # Jinja2 template path
    security_policy_filename: str = "security_policy.j2"
    system_prompt_kwargs: dict[str, object] = {}           # Extra kwargs for system prompt template
    condenser: CondenserBase | None = None                 # Context condenser
    critic: CriticBase | None = None                       # EXPERIMENTAL: Real-time evaluation
```

### Key Notes

- `tools` takes a list of `Tool` spec objects (name + params), NOT `ToolDefinition` instances. Tools are resolved from the registry at conversation init time.
- `include_default_tools` defaults to `["FinishTool", "ThinkTool"]`. Set to `[]` to disable built-in tools.
- Agent is frozen (immutable) after construction.

### Example

```python
from openhands.sdk import LLM, Agent, Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool

llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key="...")
agent = Agent(
    llm=llm,
    tools=[
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ],
)
```

---

## 3. Tool (Spec)

**Import:** `from openhands.sdk import Tool` (or `from openhands.sdk.tool import Tool`)

`Tool` is a lightweight spec that references a registered tool by name and optional params. It is NOT the tool implementation itself -- it is passed to `Agent(tools=[...])` and resolved at runtime.

### Signature

```python
class Tool(BaseModel):
    name: str       # Name of registered tool class, e.g., "TerminalTool"
    params: dict[str, Any] = {}  # Params passed to ToolDefinition.create()
```

### Usage

```python
from openhands.sdk import Tool
from openhands.tools.terminal import TerminalTool

# Use the class's .name attribute to get the registered name
tool = Tool(name=TerminalTool.name)

# With params
tool = Tool(name="TerminalTool", params={"timeout": 120})
```

---

## 4. Conversation

**Import:** `from openhands.sdk import Conversation` (or `from openhands.sdk.conversation import Conversation`)

`Conversation` is a factory class that returns either a `LocalConversation` or `RemoteConversation` based on workspace type.

### Constructor Signature

```python
class Conversation:
    def __new__(
        cls,
        agent: AgentBase,
        *,
        workspace: str | Path | LocalWorkspace | RemoteWorkspace = "workspace/project",
        plugins: list[PluginSource] | None = None,
        persistence_dir: str | Path | None = None,
        conversation_id: ConversationID | None = None,
        callbacks: list[ConversationCallbackType] | None = None,
        token_callbacks: list[ConversationTokenCallbackType] | None = None,
        hook_config: HookConfig | None = None,
        max_iteration_per_run: int = 500,
        stuck_detection: bool = True,
        stuck_detection_thresholds: StuckDetectionThresholds | Mapping[str, int] | None = None,
        visualizer: type[ConversationVisualizerBase] | ConversationVisualizerBase | None = DefaultConversationVisualizer,
        secrets: dict[str, SecretValue] | dict[str, str] | None = None,
        delete_on_close: bool = True,
    ) -> LocalConversation | RemoteConversation
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `agent` | The Agent instance |
| `workspace` | Working directory for tool execution. String/Path for local, `RemoteWorkspace` for remote. |
| `plugins` | List of `PluginSource` for loading skills, MCP tools, hooks from external repos |
| `persistence_dir` | Directory to persist conversation state. Enables resume. |
| `conversation_id` | Explicit conversation ID. Auto-generated UUID if not provided. |
| `callbacks` | List of `Callable[[Event], None]` invoked on each event |
| `token_callbacks` | Callbacks for streaming token deltas |
| `hook_config` | Hook configuration (shell scripts at lifecycle events) |
| `max_iteration_per_run` | Max agent steps per `.run()` call (default 500) |
| `stuck_detection` | Enable stuck detection (default True) |
| `visualizer` | Terminal visualization. Set to `None` to disable. |
| `secrets` | Dict of secrets available to the agent |
| `delete_on_close` | Delete workspace on close (default True) |

### Key Methods

```python
conversation.send_message("Hello!")          # Send a user message (str or Message)
conversation.send_message(message, sender="sub_agent")  # With sender tracking
conversation.run()                           # Run agent until finished
conversation.pause()                         # Pause execution
conversation.close()                         # Clean up resources
conversation.reject_pending_actions(reason)  # Reject pending actions in confirmation mode
conversation.set_confirmation_policy(policy) # Set confirmation policy
conversation.set_security_analyzer(analyzer) # Set security analyzer
conversation.generate_title(llm=None)        # Generate conversation title
conversation.update_secrets({"key": "val"})  # Update runtime secrets
```

### Key Properties

```python
conversation.id                  # ConversationID (UUID)
conversation.state               # ConversationStateProtocol
conversation.state.events        # EventsListBase (sequence of events)
conversation.conversation_stats  # ConversationStats with metrics
```

### Example

```python
import os
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.terminal import TerminalTool

llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key=os.getenv("LLM_API_KEY"))
agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name)])
conversation = Conversation(agent=agent, workspace=os.getcwd())

conversation.send_message("List files in the current directory")
conversation.run()

# Multi-turn
conversation.send_message("Now create a file called hello.txt")
conversation.run()
```

### Persistence (Resume Conversations)

```python
# Create with persistence
conversation = Conversation(
    agent=agent,
    workspace=".",
    persistence_dir="./.conversations",
)
conversation.send_message("Do something")
conversation.run()

# Later: resume by providing same persistence_dir (auto-detects existing state)
conversation = Conversation(
    agent=agent,
    workspace=".",
    persistence_dir="./.conversations",
)
conversation.send_message("Continue from where we left off")
conversation.run()
```

---

## 5. AgentContext

**Import:** `from openhands.sdk import AgentContext` (or `from openhands.sdk.context import AgentContext`)

AgentContext manages prompt extensions: skills, system/user message suffixes, secrets, and datetime context.

### Signature

```python
class AgentContext(BaseModel):
    skills: list[Skill] = []
    system_message_suffix: str | None = None   # Appended to system prompt
    user_message_suffix: str | None = None     # Appended to each user message
    load_user_skills: bool = False             # Load from ~/.openhands/skills/
    load_public_skills: bool = False           # Load from public OpenHands extensions repo
    marketplace_path: str | None = DEFAULT_MARKETPLACE_PATH  # Public skills filter
    secrets: Mapping[str, SecretValue] | None = None
    current_datetime: datetime | str | None = datetime.now()  # Time context for agent
```

### Example

```python
from openhands.sdk import AgentContext, Agent
from openhands.sdk.context import Skill, KeywordTrigger

agent_context = AgentContext(
    skills=[
        Skill(
            name="repo.md",
            content="You are working on a Python web application.",
            trigger=None,  # Always active (repo skill)
        ),
        Skill(
            name="deployment",
            content="Use docker-compose for deployment. Never use bare docker run.",
            trigger=KeywordTrigger(keywords=["deploy", "deployment", "docker"]),
        ),
    ],
    system_message_suffix="Always explain your reasoning before taking action.",
    user_message_suffix="Remember to check tests after changes.",
    load_public_skills=True,
)

agent = Agent(llm=llm, tools=tools, agent_context=agent_context)
```

---

## 6. Skill

**Import:** `from openhands.sdk.context import Skill, KeywordTrigger, TaskTrigger`

Skills provide specialized knowledge or functionality injected into prompts.

### Signature

```python
class Skill(BaseModel):
    name: str                          # Unique skill name
    content: str                       # The skill content/instructions
    trigger: KeywordTrigger | TaskTrigger | None = None  # When to activate
    source: str | None = None          # Source path or identifier
    description: str | None = None     # Brief description
    mcp_tools: dict | None = None      # MCP tools config (repo skills)
    is_agentskills_format: bool = False
    # Additional AgentSkills standard fields:
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    allowed_tools: list[str] | None = None
    resources: SkillResources | None = None
```

### Trigger Types

```python
from openhands.sdk.context import KeywordTrigger, TaskTrigger

# Keyword trigger: activated when keywords appear in user messages
KeywordTrigger(keywords=["docker", "container", "deploy"])

# Task trigger: activated for specific task types
TaskTrigger(triggers=["code_review", "refactor"])
```

### Skill Behavior

- **`trigger=None`** (repo skill): Content always included in system prompt via `<REPO_CONTEXT>`.
- **`trigger=KeywordTrigger(...)`** (knowledge skill): Listed in `<available_skills>`, content injected when keywords match user message.
- Skills are deduplicated by name. Only the first activation per conversation turn is used.

---

## 7. Custom Tool Definition Pattern

To define a custom tool, you need five components: Action, Observation, ToolExecutor, ToolDefinition subclass, and registration.

### Imports

```python
from pydantic import Field
from collections.abc import Sequence
from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
from openhands.sdk.llm import TextContent, ImageContent
```

### Step 1: Define Action (input schema)

```python
class MyAction(Action):
    """Input parameters for the tool. Fields become the tool's JSON schema."""
    query: str = Field(description="The search query")
    max_results: int = Field(default=10, description="Maximum results to return")
```

### Step 2: Define Observation (output schema)

```python
class MyObservation(Observation):
    """Output from the tool."""
    results: list[str] = Field(default_factory=list)
    count: int = 0

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """Override to customize what the LLM sees."""
        if not self.count:
            return [TextContent(text="No results found.")]
        formatted = "\n".join(f"- {r}" for r in self.results)
        return [TextContent(text=f"Found {self.count} results:\n{formatted}")]
```

### Step 3: Define Executor

```python
class MyExecutor(ToolExecutor[MyAction, MyObservation]):
    def __init__(self, some_config: str):
        self.config = some_config

    def __call__(self, action: MyAction, conversation=None) -> MyObservation:
        # Implement tool logic here
        results = [f"Result for: {action.query}"]
        return MyObservation(results=results, count=len(results))

    def close(self) -> None:
        """Optional cleanup."""
        pass
```

### Step 4: Define ToolDefinition subclass

```python
class MyTool(ToolDefinition[MyAction, MyObservation]):
    """Custom tool description shown to the LLM."""

    @classmethod
    def create(cls, conv_state, **params) -> Sequence[ToolDefinition]:
        """Factory method called at conversation init.

        Args:
            conv_state: ConversationState with workspace info.
            **params: From Tool(params={...}).
        """
        executor = MyExecutor(some_config=params.get("config", "default"))
        return [
            cls(
                description="Describe what this tool does for the LLM.",
                action_type=MyAction,
                observation_type=MyObservation,
                executor=executor,
            )
        ]
```

### Step 5: Register and Use

```python
# Register the tool (must happen before Agent construction)
register_tool("MyTool", MyTool)

# Use in agent
agent = Agent(
    llm=llm,
    tools=[
        Tool(name="MyTool", params={"config": "custom_value"}),
    ],
)
```

### Alternative: Register a Factory Function

```python
def make_tools(conv_state, **params) -> list[ToolDefinition]:
    """Factory function that creates multiple tools sharing resources."""
    executor_a = ...
    executor_b = ...
    return [tool_a, tool_b]

register_tool("MyToolSet", make_tools)
# Then: Tool(name="MyToolSet")
```

### ToolDefinition Key Fields

```python
class ToolDefinition[ActionT, ObservationT](ABC):
    name: ClassVar[str]           # Auto-set from class name (CamelCase -> snake_case, minus _tool suffix)
    description: str              # Shown to LLM
    action_type: type[Action]     # Input schema class
    observation_type: type[Observation] | None  # Output schema class (optional)
    annotations: ToolAnnotations | None = None  # MCP-style hints
    executor: ToolExecutor | None = None        # The callable executor
    meta: dict[str, Any] | None = None
```

### ToolAnnotations

```python
class ToolAnnotations(BaseModel):
    title: str | None = None
    readOnlyHint: bool = False        # If True, tool does not modify environment
    destructiveHint: bool = True      # If True, tool may perform destructive updates
    idempotentHint: bool = False
    openWorldHint: bool = True        # If True, interacts with external entities
```

### Overriding the Built-in FinishTool

To replace the default `FinishTool` with a custom one (e.g., to enforce a structured output schema), define a `ToolDefinition` subclass named exactly `FinishTool` and register it under that name.

**Key rules:**
1. The class must be named `FinishTool` so it replaces the built-in.
2. Remove `"FinishTool"` from `include_default_tools` on the Agent — keep only `"ThinkTool"` (or an empty list).
3. Add `Tool(name="FinishTool")` to the agent's `tools` list explicitly.
4. **CRITICAL — Action class names must be unique.** Do NOT name your Action class `FinishAction`. The SDK registers all Action/Observation subclasses globally in a Pydantic discriminated union. Reusing the built-in names causes a "Duplicate class definition" error that breaks validation of every event in the conversation. Use a distinct name such as `StructuredFinishAction`.
5. **Do not define custom FinishObservation class unless necessary.** By default, `FinishObservation`just echos what models write through `FinishAction`. That will be the desired behavior, as `FinishAction` often contains information we want to extract.

---

## 8. Condensers

Condensers manage conversation context by summarizing or trimming event history when it grows too large.

**Import:** `from openhands.sdk.context.condenser import LLMSummarizingCondenser, NoOpCondenser, PipelineCondenser`

Also re-exported: `from openhands.sdk import LLMSummarizingCondenser`

### LLMSummarizingCondenser

The primary condenser. Uses an LLM to summarize forgotten events when conversation exceeds size limits.

```python
class LLMSummarizingCondenser(RollingCondenser):
    llm: LLM                          # LLM for generating summaries
    max_size: int = 240                # Max events before condensation triggers
    max_tokens: int | None = None      # Optional token-based limit
    keep_first: int = 2                # Number of initial events to preserve (system prompt, first message)
    minimum_progress: float = 0.1      # Min fraction of events that must be condensed (0.0-1.0)
    hard_context_reset_max_retries: int = 5
    hard_context_reset_context_scaling: float = 0.8
```

**Example:**

```python
from openhands.sdk import LLM, Agent, LLMSummarizingCondenser

condenser = LLMSummarizingCondenser(
    llm=llm.model_copy(update={"usage_id": "condenser"}),
    max_size=10,
    keep_first=2,
)

agent = Agent(llm=llm, tools=tools, condenser=condenser)
```

### NoOpCondenser

Returns the view unchanged. For testing.

```python
class NoOpCondenser(CondenserBase):
    pass  # condense() returns view unchanged
```

### PipelineCondenser

Chains multiple condensers in sequence.

```python
class PipelineCondenser(CondenserBase):
    condensers: list[CondenserBase]   # Applied in order; stops on Condensation
```

**Example:**

```python
from openhands.sdk.context.condenser import PipelineCondenser, LLMSummarizingCondenser

pipeline = PipelineCondenser(condensers=[
    LLMSummarizingCondenser(llm=llm, max_size=100),
])
agent = Agent(llm=llm, tools=tools, condenser=pipeline)
```

### CondenserBase (Abstract)

```python
class CondenserBase(ABC):
    def condense(self, view: View, agent_llm: LLM | None = None) -> View | Condensation: ...
    def handles_condensation_requests(self) -> bool: ...  # Default: False
```

---

## 9. Hooks

Hooks are shell scripts that execute at specific lifecycle events during agent execution.

**Import:** `from openhands.sdk.hooks import HookConfig, HookDefinition, HookMatcher, HookType`

### HookConfig

```python
class HookConfig(BaseModel):
    pre_tool_use: list[HookMatcher] = []       # Before tool execution
    post_tool_use: list[HookMatcher] = []      # After tool execution
    user_prompt_submit: list[HookMatcher] = [] # When user submits a prompt
    session_start: list[HookMatcher] = []      # When session starts
    session_end: list[HookMatcher] = []        # When session ends
    stop: list[HookMatcher] = []               # When agent attempts to stop
```

### HookMatcher

```python
class HookMatcher(BaseModel):
    matcher: str = "*"                   # Tool name pattern: "*" (wildcard), exact name, or regex
    hooks: list[HookDefinition] = []
```

### HookDefinition

```python
class HookDefinition(BaseModel):
    type: HookType = HookType.COMMAND    # "command" (shell) or "prompt" (future)
    command: str                         # Shell command to execute
    timeout: int = 60                    # Timeout in seconds
    async_: bool = False                 # (alias: "async") Run asynchronously
```

### HookEvent (stdin JSON to scripts)

```python
class HookEvent(BaseModel):
    event_type: HookEventType     # "PreToolUse", "PostToolUse", etc.
    tool_name: str | None
    tool_input: dict | None
    tool_response: dict | None
    message: str | None
    session_id: str | None
    working_dir: str | None
    metadata: dict[str, Any] = {}
```

### HookDecision (stdout JSON from scripts)

Scripts output JSON with `{"decision": "allow"}` or `{"decision": "deny", "reason": "..."}`.

### Example

```python
from openhands.sdk.hooks import HookConfig, HookMatcher, HookDefinition

hook_config = HookConfig(
    pre_tool_use=[
        HookMatcher(
            matcher="terminal",
            hooks=[
                HookDefinition(command="/path/to/block_dangerous.sh", timeout=10)
            ],
        )
    ],
    post_tool_use=[
        HookMatcher(
            matcher="*",
            hooks=[
                HookDefinition(command="/path/to/log_tools.sh", timeout=5)
            ],
        )
    ],
    stop=[
        HookMatcher(
            hooks=[
                HookDefinition(command="/path/to/require_summary.sh")
            ],
        )
    ],
)

conversation = Conversation(agent=agent, workspace=".", hook_config=hook_config)
```

### Loading from JSON

```python
# From file
config = HookConfig.load(".openhands/hooks.json")

# From dict (supports PascalCase keys)
config = HookConfig.from_dict({
    "hooks": {
        "PreToolUse": [{"matcher": "terminal", "hooks": [{"command": "script.sh"}]}]
    }
})

# Merge multiple configs
merged = HookConfig.merge([config1, config2])
```

---

## 10. Built-in Tools

Tools are in the `openhands.tools` package. Each tool module exports a `*Tool` class with a `.name` class attribute.

| Tool Class | Import | `.name` value | Description |
|------------|--------|---------------|-------------|
| `TerminalTool` | `from openhands.tools.terminal import TerminalTool` | `"terminal"` | Shell command execution |
| `FileEditorTool` | `from openhands.tools.file_editor import FileEditorTool` | `"file_editor"` | File viewing/editing |
| `TaskTrackerTool` | `from openhands.tools.task_tracker import TaskTrackerTool` | `"task_tracker"` | Task tracking |
| `GrepTool` | `from openhands.tools.grep import GrepTool` | `"grep"` | Regex content search |
| `GlobTool` | `from openhands.tools.glob import GlobTool` | `"glob"` | File pattern matching |
| `ApplyPatchTool` | `from openhands.tools.apply_patch import ApplyPatchTool` | `"apply_patch"` | Apply patch files |

### Built-in Default Tools (always included unless overridden)

| Tool Class | `.name` value | Description |
|------------|---------------|-------------|
| `FinishTool` | `"finish"` | Signals task completion |
| `ThinkTool` | `"think"` | Internal reasoning step |

These are controlled by `Agent(include_default_tools=["FinishTool", "ThinkTool"])`.

### Using Tools

```python
from openhands.sdk import Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool

tools = [
    Tool(name=TerminalTool.name),     # "terminal"
    Tool(name=FileEditorTool.name),   # "file_editor"
]
```

### Preset Agents

Pre-configured agent factories with standard tool sets:

```python
from openhands.tools.preset.default import get_default_agent
from openhands.tools.preset.planning import get_planning_agent

# Default agent with terminal, file editor, task tracker, etc.
agent = get_default_agent(llm=llm, cli_mode=True)

# Planning agent with read-only tools
planning_agent = get_planning_agent(llm=llm)
```

---

## 11. MCP Integration

MCP (Model Context Protocol) tools can be added via the `mcp_config` parameter on Agent.

```python
mcp_config = {
    "mcpServers": {
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
        "repomix": {"command": "npx", "args": ["-y", "repomix@1.4.2", "--mcp"]},
    }
}

agent = Agent(
    llm=llm,
    tools=[Tool(name=TerminalTool.name)],
    mcp_config=mcp_config,
    # Optional: filter MCP tools by regex
    filter_tools_regex="^(?!repomix)(.*)|^repomix.*pack_codebase.*$",
)
```

---

## 12. Agent Delegation

Sub-agents can be registered and delegated to via the `DelegateTool`.

**Imports:**

```python
from openhands.sdk.subagent import register_agent
from openhands.sdk.tool import register_tool
from openhands.tools.delegate import DelegateTool, DelegationVisualizer
from openhands.tools import register_builtins_agents
```

### Register Sub-Agents

```python
def create_specialist(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        tools=[],
        agent_context=AgentContext(
            skills=[Skill(name="spec", content="...", trigger=None)],
        ),
    )

register_agent(
    name="specialist",
    factory_func=create_specialist,
    description="A specialist agent for X.",
)
register_builtins_agents()  # Register default agent types
register_tool("DelegateTool", DelegateTool)
```

### Use Delegation

```python
main_agent = Agent(llm=llm, tools=[Tool(name="DelegateTool")])
conversation = Conversation(
    agent=main_agent,
    workspace=os.getcwd(),
    visualizer=DelegationVisualizer(name="Delegator"),
)
conversation.send_message("Delegate task X to the specialist agent")
conversation.run()
```

---

## 13. Critic (Experimental)

Critics evaluate agent actions/messages in real-time for iterative refinement.

**Import:** `from openhands.sdk.critic import APIBasedCritic, IterativeRefinementConfig`

```python
from openhands.sdk.critic import APIBasedCritic, IterativeRefinementConfig

iterative_config = IterativeRefinementConfig(
    success_threshold=0.7,   # Score threshold for success
    max_iterations=3,        # Max retry iterations
)

critic = APIBasedCritic(
    server_url="https://my-critic-server.com",
    api_key="my-api-key",
    model_name="critic",
    iterative_refinement=iterative_config,
)

agent = Agent(llm=llm, tools=tools, critic=critic)
conversation = Conversation(agent=agent, workspace=".")
conversation.send_message("Build X")
conversation.run()  # Automatically retries based on critic scores
```

---

## 14. Events and Callbacks

Callbacks receive `Event` objects for each step of the conversation.

**Import:** `from openhands.sdk import Event, LLMConvertibleEvent, MessageEvent`

### Callback Signature

```python
from openhands.sdk import Event, LLMConvertibleEvent

def my_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_message = event.to_llm_message()  # Get the Message object
        # Process...

conversation = Conversation(agent=agent, workspace=".", callbacks=[my_callback])
```

### Key Event Types

| Event | Import | Description |
|-------|--------|-------------|
| `MessageEvent` | `from openhands.sdk.event import MessageEvent` | User or agent message |
| `ActionEvent` | `from openhands.sdk.event import ActionEvent` | Agent tool call |
| `ObservationEvent` | `from openhands.sdk.event import ObservationEvent` | Tool result |
| `SystemPromptEvent` | `from openhands.sdk.event import SystemPromptEvent` | System prompt |
| `AgentErrorEvent` | `from openhands.sdk.event import AgentErrorEvent` | Tool validation error |

### Thinking Blocks (Anthropic Extended Thinking)

```python
from openhands.sdk import ThinkingBlock, RedactedThinkingBlock

def show_thinking(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        msg = event.to_llm_message()
        if hasattr(msg, "thinking_blocks") and msg.thinking_blocks:
            for block in msg.thinking_blocks:
                if isinstance(block, ThinkingBlock):
                    print(f"Thinking: {block.thinking}")
                elif isinstance(block, RedactedThinkingBlock):
                    print(f"Redacted: {block.data}")
```

---

## 15. LLM Registry

Track and reuse LLM instances by usage ID.

**Import:** `from openhands.sdk import LLMRegistry`

```python
from openhands.sdk import LLM, LLMRegistry

registry = LLMRegistry()
registry.add(LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key="...", usage_id="agent"))
registry.add(LLM(model="anthropic/claude-haiku-4-5-20251001", api_key="...", usage_id="cheap"))

agent_llm = registry.get("agent")
cheap_llm = registry.get("cheap")

# List all registered
print(registry.list_usage_ids())
```

---

## 16. Common Patterns

### Minimal Agent

```python
import os
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool

llm = LLM(
    model=os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-5-20250929"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL", None),
)
agent = Agent(
    llm=llm,
    tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)],
)
conversation = Conversation(agent=agent, workspace=os.getcwd())
conversation.send_message("Your task here")
conversation.run()
```

### Agent with Context Condenser

```python
from openhands.sdk import LLM, Agent, Conversation, Tool, LLMSummarizingCondenser
from openhands.tools.terminal import TerminalTool

llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key="...")
condenser = LLMSummarizingCondenser(
    llm=llm.model_copy(update={"usage_id": "condenser"}),
    max_size=80,
    keep_first=2,
)
agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name)], condenser=condenser)
conversation = Conversation(agent=agent, workspace=".", persistence_dir="./.conversations")
```

### Two-Phase Workflow (Planning + Execution)

```python
from openhands.sdk import LLM, Conversation
from openhands.tools.preset.planning import get_planning_agent
from openhands.tools.preset.default import get_default_agent

llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key="...")

# Phase 1: Plan
planning_agent = get_planning_agent(llm=llm)
plan_conv = Conversation(agent=planning_agent, workspace="/workspace")
plan_conv.send_message("Analyze and plan: <task>")
plan_conv.run()

# Phase 2: Execute
exec_agent = get_default_agent(llm=llm, cli_mode=True)
exec_conv = Conversation(agent=exec_agent, workspace="/workspace")
exec_conv.send_message("Implement the plan in PLAN.md")
exec_conv.run()
```

### Collecting LLM Messages

```python
from openhands.sdk import Event, LLMConvertibleEvent

llm_messages = []

def collect_messages(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())

conversation = Conversation(agent=agent, workspace=".", callbacks=[collect_messages])
```

### Accessing Cost

```python
# From LLM directly
cost = llm.metrics.accumulated_cost

# From conversation stats (includes all LLMs used)
cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
```

### No Visualization (Headless)

```python
conversation = Conversation(agent=agent, workspace=".", visualizer=None)
```

### Custom System Prompt

```python
agent = Agent(
    llm=llm,
    tools=tools,
    system_prompt_filename="/absolute/path/to/custom_prompt.j2",
    system_prompt_kwargs={"cli_mode": True, "repo_name": "my-project"},
)
```
