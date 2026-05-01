# 9. Hooks

**Import:** `from openhands.sdk.hooks import HookConfig, HookDefinition, HookMatcher, HookType`

Hooks let you run a shell command at specific lifecycle events. The main thing you need to define is a helper that returns a `HookConfig`. The function will later be passed to the conversation constructor and can use the workspace directory to set up hook scripts or other resources.

## Example: Enforcing a Pre-Stop Check

```python
def get_hook_config(workspace_dir: str) -> HookConfig:
    hook_script = os.path.join(workspace_dir, "stop_hook.py")
    return HookConfig(
        pre_tool_use=[
            HookMatcher(
                matcher="finish",
                hooks=[
                    HookDefinition(
                        command=f"python3 {hook_script}",
                        timeout=120,
                    )
                ]
            )
        ]
    )
```

Scripts return JSON such as `{"decision": "allow"}` or `{"decision": "deny", "reason": "..."}`.

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
)
```