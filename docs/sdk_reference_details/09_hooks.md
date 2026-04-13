# 9. Hooks

**Import:** `from openhands.sdk.hooks import HookConfig, HookDefinition, HookMatcher, HookType`

Hooks are shell scripts that run at specific lifecycle events.

## HookConfig

```python
class HookConfig(BaseModel):
    pre_tool_use: list[HookMatcher] = []
    post_tool_use: list[HookMatcher] = []
    user_prompt_submit: list[HookMatcher] = []
    session_start: list[HookMatcher] = []
    session_end: list[HookMatcher] = []
    stop: list[HookMatcher] = []
```

## HookMatcher

```python
class HookMatcher(BaseModel):
    matcher: str = "*"
    hooks: list[HookDefinition] = []
```

## HookDefinition

```python
class HookDefinition(BaseModel):
    type: HookType = HookType.COMMAND
    command: str
    timeout: int = 60
    async_: bool = False
```

## HookEvent

```python
class HookEvent(BaseModel):
    event_type: HookEventType
    tool_name: str | None
    tool_input: dict | None
    tool_response: dict | None
    message: str | None
    session_id: str | None
    working_dir: str | None
    metadata: dict[str, Any] = {}
```

Scripts return JSON such as `{"decision": "allow"}` or `{"decision": "deny", "reason": "..."}`.

## Example

```python
from openhands.sdk.hooks import HookConfig, HookDefinition, HookMatcher

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
```

## Loading from JSON

```python
config = HookConfig.load(".openhands/hooks.json")

config = HookConfig.from_dict({
    "hooks": {
        "PreToolUse": [{"matcher": "terminal", "hooks": [{"command": "script.sh"}]}]
    }
})

merged = HookConfig.merge([config1, config2])
```
