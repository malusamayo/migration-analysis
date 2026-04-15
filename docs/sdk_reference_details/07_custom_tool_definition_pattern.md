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

## Replacing the Built-in FinishTool

To replace the default `FinishTool`, follow the above steps to define your custom tool, but make sure that:

1. Use a unique action class name; do not reuse `FinishAction`.
2. Only define a custom observation type if you actually need one.
3. The `ToolDefinition` subclass is named exactly `FinishTool`. Define the `create` method to return a single instance of `FinishTool` with your custom executor.
4. Remove `"FinishTool"` from `include_default_tools` and add `Tool(name="FinishTool")` with your custom implementation to the agent's `tools` list.


# Alternative: Deploying Helper Scripts via get_workspace_scripts

`agent.py` can optionally define `get_workspace_scripts() -> dict[str, str]` to provision files into the agent's working directory at runtime. The keys are relative file paths and the values are the file contents.

```python
def get_workspace_scripts() -> dict[str, str]:
    return {
        "helpers/parse_output.py": """\
import sys, json

data = json.load(sys.stdin)
print(data["result"])
""",
    }
```

These scripts are written to the agent's workspace before execution, so the agent can call them via the terminal tool. This is useful for injecting small utilities the agent can rely on without requiring them to be installed globally.