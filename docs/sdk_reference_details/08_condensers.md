# 8. Condensers

**Import:** `from openhands.sdk.context.condenser import LLMSummarizingCondenser, NoOpCondenser, PipelineCondenser`

Condensers summarize or trim conversation history when it becomes too large.

## LLMSummarizingCondenser

```python
class LLMSummarizingCondenser(RollingCondenser):
    llm: LLM
    max_size: int = 240
    max_tokens: int | None = None
    keep_first: int = 2
    minimum_progress: float = 0.1
    hard_context_reset_max_retries: int = 5
    hard_context_reset_context_scaling: float = 0.8
```

```python
from openhands.sdk import LLM, Agent, LLMSummarizingCondenser

condenser = LLMSummarizingCondenser(
    llm=llm.model_copy(update={"usage_id": "condenser"}),
    max_size=10,
    keep_first=2,
)

agent = Agent(llm=llm, tools=tools, condenser=condenser)
```

## NoOpCondenser

```python
class NoOpCondenser(CondenserBase):
    pass
```

## PipelineCondenser

```python
class PipelineCondenser(CondenserBase):
    condensers: list[CondenserBase]
```

```python
from openhands.sdk.context.condenser import LLMSummarizingCondenser, PipelineCondenser

pipeline = PipelineCondenser(condensers=[
    LLMSummarizingCondenser(llm=llm, max_size=100),
])
agent = Agent(llm=llm, tools=tools, condenser=pipeline)
```

## CondenserBase

```python
class CondenserBase(ABC):
    def condense(self, view: View, agent_llm: LLM | None = None) -> View | Condensation: ...
    def handles_condensation_requests(self) -> bool: ...
```
