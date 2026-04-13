# 4. Conversation

**Import:** `from openhands.sdk import Conversation` or `from openhands.sdk.conversation import Conversation`

`Conversation` is a factory that returns a `LocalConversation` or `RemoteConversation` based on the workspace.

## Constructor Signature

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

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `agent` | The `Agent` instance. |
| `workspace` | Working directory for tool execution. |
| `plugins` | External plugin sources for skills, hooks, and MCP tools. |
| `persistence_dir` | Directory used to persist state for resume. |
| `conversation_id` | Explicit conversation ID. |
| `callbacks` | Event callbacks. |
| `token_callbacks` | Streaming token callbacks. |
| `hook_config` | Hook configuration. |
| `max_iteration_per_run` | Maximum agent steps per `run()`. |
| `stuck_detection` | Enables stuck detection. |
| `visualizer` | Terminal visualization; set `None` to disable. |
| `secrets` | Runtime secrets exposed to the agent. |
| `delete_on_close` | Whether to remove the workspace on close. |

## Key Methods

```python
conversation.send_message("Hello!")
conversation.send_message(message, sender="sub_agent")
conversation.run()
conversation.pause()
conversation.close()
conversation.reject_pending_actions(reason)
conversation.set_confirmation_policy(policy)
conversation.set_security_analyzer(analyzer)
conversation.generate_title(llm=None)
conversation.update_secrets({"key": "val"})
```

## Key Properties

```python
conversation.id
conversation.state
conversation.state.events
conversation.conversation_stats
```

## Example

```python
import os
from openhands.sdk import LLM, Agent, Conversation, Tool
from openhands.tools.terminal import TerminalTool

llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key=os.getenv("LLM_API_KEY"))
agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name)])
conversation = Conversation(agent=agent, workspace=os.getcwd())

conversation.send_message("List files in the current directory")
conversation.run()

conversation.send_message("Now create a file called hello.txt")
conversation.run()
```

## Persistence

```python
conversation = Conversation(
    agent=agent,
    workspace=".",
    persistence_dir="./.conversations",
)
conversation.send_message("Do something")
conversation.run()

conversation = Conversation(
    agent=agent,
    workspace=".",
    persistence_dir="./.conversations",
)
conversation.send_message("Continue from where we left off")
conversation.run()
```
