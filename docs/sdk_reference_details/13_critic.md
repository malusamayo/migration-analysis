# 13. Critic (Experimental)

**Import:** `from openhands.sdk.critic import APIBasedCritic, IterativeRefinementConfig`

Critics evaluate agent actions and can trigger retries through iterative refinement.

```python
from openhands.sdk.critic import APIBasedCritic, IterativeRefinementConfig

iterative_config = IterativeRefinementConfig(
    success_threshold=0.7,
    max_iterations=3,
)

critic = APIBasedCritic(
    server_url="https://my-critic-server.com",
    api_key="my-api-key",
    model_name="critic",
    iterative_refinement=iterative_config,
)

agent = Agent(llm=llm, tools=tools, critic=critic)
conversation = Conversation(agent=agent, workspace=".")
conversation.send_message("Build X")
conversation.run()
```
