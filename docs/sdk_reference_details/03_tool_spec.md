# 3. Tool (Spec)

**Import:** `from openhands.sdk import Tool` or `from openhands.sdk.tool import Tool`

`Tool` is a lightweight spec that points to a registered tool by name and optional creation params.

## Signature

```python
class Tool(BaseModel):
    name: str
    params: dict[str, Any] = {}
```

## Usage

```python
from openhands.sdk import Tool
from openhands.tools.terminal import TerminalTool

tool = Tool(name=TerminalTool.name)
tool = Tool(name="TerminalTool", params={"timeout": 120})
```
