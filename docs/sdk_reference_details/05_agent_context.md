# 5. AgentContext

**Import:** `from openhands.sdk import AgentContext` or `from openhands.sdk.context import AgentContext`

`AgentContext` manages prompt extensions, skills, secrets, and time context.

## Signature

```python
class AgentContext(BaseModel):
    skills: list[Skill] = []
    system_message_suffix: str | None = None
    user_message_suffix: str | None = None
    load_user_skills: bool = False
    load_public_skills: bool = False
    marketplace_path: str | None = DEFAULT_MARKETPLACE_PATH
    secrets: Mapping[str, SecretValue] | None = None
    current_datetime: datetime | str | None = datetime.now()
```

## Example

```python
from openhands.sdk import AgentContext, Agent
from openhands.sdk.context import Skill, KeywordTrigger

agent_context = AgentContext(
    skills=[
        Skill(
            name="repo.md",
            content="You are working on a Python web application.",
            trigger=None,
        ),
        Skill(
            name="deployment",
            content="Use docker-compose for deployment. Never use bare docker run.",
            trigger=KeywordTrigger(keywords=["deploy", "deployment", "docker"]),
        ),
    ],
    system_message_suffix="Always explain your reasoning before taking action.",
    user_message_suffix="Remember to check tests after changes.",
    load_public_skills=True,
)

agent = Agent(llm=llm, tools=tools, agent_context=agent_context)
```
