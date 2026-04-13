# 10. Built-in Tools

Built-in tools live under `openhands.tools`.

## Standard Tool Catalog

| Tool Class | Import | `.name` value | Description |
|------------|--------|---------------|-------------|
| `TerminalTool` | `from openhands.tools.terminal import TerminalTool` | `"terminal"` | Shell command execution |
| `FileEditorTool` | `from openhands.tools.file_editor import FileEditorTool` | `"file_editor"` | File viewing and editing |
| `TaskTrackerTool` | `from openhands.tools.task_tracker import TaskTrackerTool` | `"task_tracker"` | Task tracking |
| `GrepTool` | `from openhands.tools.grep import GrepTool` | `"grep"` | Regex content search |
| `GlobTool` | `from openhands.tools.glob import GlobTool` | `"glob"` | File pattern matching |
| `ApplyPatchTool` | `from openhands.tools.apply_patch import ApplyPatchTool` | `"apply_patch"` | Apply patch files |

## Default Tools

| Tool Class | `.name` value | Description |
|------------|---------------|-------------|
| `FinishTool` | `"finish"` | Signals task completion |
| `ThinkTool` | `"think"` | Internal reasoning step |

These are controlled by `Agent(include_default_tools=["FinishTool", "ThinkTool"])`.

## Using Tools

```python
from openhands.sdk import Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

tools = [
    Tool(name=TerminalTool.name),
    Tool(name=FileEditorTool.name),
]
```

## Preset Agents

```python
from openhands.tools.preset.default import get_default_agent
from openhands.tools.preset.planning import get_planning_agent

agent = get_default_agent(llm=llm, cli_mode=True)
planning_agent = get_planning_agent(llm=llm)
```
