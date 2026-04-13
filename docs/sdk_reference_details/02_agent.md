# 2. Agent

**Import:** `from openhands.sdk import Agent` or `from openhands.sdk.agent import Agent`

The `Agent` class is the SDK's core orchestrator. It is configured with an `LLM`, tool specs, optional prompt context, optional MCP servers, and optional runtime helpers such as condensers and critics.

## Constructor Signature

```python
class Agent(CriticMixin, AgentBase):
    llm: LLM
    tools: list[Tool] = []
    mcp_config: dict[str, Any] = {}
    filter_tools_regex: str | None = None
    include_default_tools: list[str] = ["FinishTool", "ThinkTool"]
    agent_context: AgentContext | None = None
    system_prompt_filename: str = "system_prompt.j2"
    security_policy_filename: str = "security_policy.j2"
    system_prompt_kwargs: dict[str, object] = {}
    condenser: CondenserBase | None = None
    critic: CriticBase | None = None
```

## Key Notes

- `tools` accepts `Tool` specs, not `ToolDefinition` instances.
- `include_default_tools` defaults to `["FinishTool", "ThinkTool"]`; set it to `[]` to remove the built-ins.
- Agents are frozen after construction.

## Example

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
