# 12. Agent Delegation

Sub-agents can be registered and delegated to through `DelegateTool`.

## Imports

```python
from openhands.sdk.subagent import register_agent
from openhands.sdk.tool import register_tool
from openhands.tools import register_builtins_agents
from openhands.tools.delegate import DelegateTool, DelegationVisualizer
```

## Register Sub-Agents

```python
def create_specialist(llm: LLM) -> Agent:
    return Agent(
        llm=llm,
        tools=[],
        agent_context=AgentContext(
            skills=[Skill(name="spec", content="...", trigger=None)],
        ),
    )

register_agent(
    name="specialist",
    factory_func=create_specialist,
    description="A specialist agent for X.",
)
register_builtins_agents()
register_tool("DelegateTool", DelegateTool)
```

## Use Delegation

```python
main_agent = Agent(llm=llm, tools=[Tool(name="DelegateTool")])
conversation = Conversation(
    agent=main_agent,
    workspace=os.getcwd(),
    visualizer=DelegationVisualizer(name="Delegator"),
)
conversation.send_message("Delegate task X to the specialist agent")
conversation.run()
```
