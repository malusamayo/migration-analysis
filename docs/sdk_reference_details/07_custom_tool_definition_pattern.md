# 7. Custom Tool Definition Pattern

Custom tools are built from five parts: `Action`, `Observation`, `ToolExecutor`, `ToolDefinition`, and registration.

## Imports

```python
from collections.abc import Sequence
from pydantic import Field

from openhands.sdk import Action, Observation, ToolDefinition
from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.tool import Tool, ToolExecutor, register_tool
```

## Step 1: Define Action

```python
class MyAction(Action):
    query: str = Field(description="The search query")
    max_results: int = Field(default=10, description="Maximum results to return")
```

## Step 2: Define Observation

```python
class MyObservation(Observation):
    results: list[str] = Field(default_factory=list)
    count: int = 0

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        if not self.count:
            return [TextContent(text="No results found.")]
        formatted = "\n".join(f"- {r}" for r in self.results)
        return [TextContent(text=f"Found {self.count} results:\n{formatted}")]
```

## Step 3: Define Executor

```python
class MyExecutor(ToolExecutor[MyAction, MyObservation]):
    def __init__(self, some_config: str):
        self.config = some_config

    def __call__(self, action: MyAction, conversation=None) -> MyObservation:
        results = [f"Result for: {action.query}"]
        return MyObservation(results=results, count=len(results))

    def close(self) -> None:
        pass
```

## Step 4: Define ToolDefinition

```python
class MyTool(ToolDefinition[MyAction, MyObservation]):
    @classmethod
    def create(cls, conv_state, **params) -> Sequence[ToolDefinition]:
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

## Step 5: Register and Use

```python
register_tool("MyTool", MyTool)

agent = Agent(
    llm=llm,
    tools=[
        Tool(name="MyTool", params={"config": "custom_value"}),
    ],
)
```

## Alternative: Register a Factory Function

```python
def make_tools(conv_state, **params) -> list[ToolDefinition]:
    executor_a = ...
    executor_b = ...
    return [tool_a, tool_b]

register_tool("MyToolSet", make_tools)
```

## ToolDefinition Key Fields

```python
class ToolDefinition[ActionT, ObservationT](ABC):
    name: ClassVar[str]
    description: str
    action_type: type[Action]
    observation_type: type[Observation] | None
    annotations: ToolAnnotations | None = None
    executor: ToolExecutor | None = None
    meta: dict[str, Any] | None = None
```

## ToolAnnotations

```python
class ToolAnnotations(BaseModel):
    title: str | None = None
    readOnlyHint: bool = False
    destructiveHint: bool = True
    idempotentHint: bool = False
    openWorldHint: bool = True
```

## Replacing the Built-in FinishTool

To replace the default `FinishTool`:

1. Define a `ToolDefinition` subclass named exactly `FinishTool`.
2. Remove `"FinishTool"` from `include_default_tools`.
3. Add `Tool(name="FinishTool")` explicitly to the agent's `tools`.
4. Use a unique action class name; do not reuse `FinishAction`.
5. Only define a custom observation type if you actually need one.
