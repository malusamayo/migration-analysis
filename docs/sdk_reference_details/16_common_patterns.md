# 16. Common Patterns

## Minimal Agent

```python
import os
from openhands.sdk import Agent, Conversation, LLM, Tool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool

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

## Agent with Context Condenser

```python
from openhands.sdk import Agent, Conversation, LLM, LLMSummarizingCondenser, Tool
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

## Two-Phase Workflow

```python
from openhands.sdk import Conversation, LLM
from openhands.tools.preset.default import get_default_agent
from openhands.tools.preset.planning import get_planning_agent

llm = LLM(model="anthropic/claude-sonnet-4-5-20250929", api_key="...")

planning_agent = get_planning_agent(llm=llm)
plan_conv = Conversation(agent=planning_agent, workspace="/workspace")
plan_conv.send_message("Analyze and plan: <task>")
plan_conv.run()

exec_agent = get_default_agent(llm=llm, cli_mode=True)
exec_conv = Conversation(agent=exec_agent, workspace="/workspace")
exec_conv.send_message("Implement the plan in PLAN.md")
exec_conv.run()
```

## Collecting LLM Messages

```python
from openhands.sdk import Event, LLMConvertibleEvent

llm_messages = []

def collect_messages(event: Event):
    if isinstance(event, LLMConvertibleEvent):
        llm_messages.append(event.to_llm_message())

conversation = Conversation(agent=agent, workspace=".", callbacks=[collect_messages])
```

## Accessing Cost

```python
cost = llm.metrics.accumulated_cost
cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
```

## Headless Mode

```python
conversation = Conversation(agent=agent, workspace=".", visualizer=None)
```

## Custom System Prompt

```python
agent = Agent(
    llm=llm,
    tools=tools,
    system_prompt_filename="/absolute/path/to/custom_prompt.j2",
    system_prompt_kwargs={"cli_mode": True, "repo_name": "my-project"},
)
```
