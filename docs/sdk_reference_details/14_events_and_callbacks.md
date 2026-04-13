# 14. Events and Callbacks

**Import:** `from openhands.sdk import Event, LLMConvertibleEvent, MessageEvent`

Callbacks receive `Event` objects as the conversation progresses.

## Callback Signature

```python
from openhands.sdk import Event, LLMConvertibleEvent

def my_callback(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_message = event.to_llm_message()

conversation = Conversation(agent=agent, workspace=".", callbacks=[my_callback])
```

## Key Event Types

| Event | Import | Description |
|-------|--------|-------------|
| `MessageEvent` | `from openhands.sdk.event import MessageEvent` | User or agent message |
| `ActionEvent` | `from openhands.sdk.event import ActionEvent` | Agent tool call |
| `ObservationEvent` | `from openhands.sdk.event import ObservationEvent` | Tool result |
| `SystemPromptEvent` | `from openhands.sdk.event import SystemPromptEvent` | System prompt |
| `AgentErrorEvent` | `from openhands.sdk.event import AgentErrorEvent` | Tool validation error |

## Thinking Blocks

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
