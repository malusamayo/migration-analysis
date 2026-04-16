# OpenHands Agent SDK Reference

This document is the high-level map to the OpenHands Agent SDK. It keeps the numbered concepts in one place and points to companion markdown files for signatures, examples, and implementation details.

Implementation details live in [`docs/sdk_reference_details/`](sdk_reference_details/).

---

## Table of Contents

1. [LLM](#1-llm)
2. [Agent](#2-agent)
3. [Tool (Spec)](#3-tool-spec)
4. [Conversation](#4-conversation)
5. [AgentContext](#5-agentcontext)
6. [Skill](#6-skill)
7. [Custom Tool Definition Pattern](#7-custom-tool-definition-pattern)
8. [Condensers](#8-condensers)
9. [Hooks](#9-hooks)
10. [Built-in Tools](#10-built-in-tools)
11. [MCP Integration](#11-mcp-integration)
12. [Agent Delegation](#12-agent-delegation)
13. [Critic (Experimental)](#13-critic-experimental)
14. [Events and Callbacks](#14-events-and-callbacks)
15. [LLM Registry](#15-llm-registry)
16. [Common Patterns](#16-common-patterns)
17. [Retriever Tool](#17-retriever-tool)

---

## 1. LLM

`LLM` is the SDK's model client. It centralizes provider selection, transport settings, retry behavior, streaming, and usage accounting so the rest of the SDK can treat model access uniformly.

Use this section when you need to configure model access or reuse the same model client across agents and other SDK components.

Implementation details: [`sdk_reference_details/01_llm.md`](sdk_reference_details/01_llm.md)

---

## 2. Agent

`Agent` is the main orchestration object. It combines an `LLM` with tools, prompt context, optional MCP servers, optional context condensation, and optional critique so a conversation can execute a task.

Use this section when you are deciding how an agent is assembled and which runtime behaviors belong on the agent rather than on the conversation.

Implementation details: [`sdk_reference_details/02_agent.md`](sdk_reference_details/02_agent.md)

---

## 3. Tool (Spec)

`Tool` is the lightweight declaration used to tell an `Agent` which registered tools to resolve at runtime. It is a reference to a tool implementation, not the implementation itself.

Use this section when you want to add an existing tool to an agent or pass creation parameters into a registered tool factory.

Implementation details: [`sdk_reference_details/03_tool_spec.md`](sdk_reference_details/03_tool_spec.md)

---

## 4. Conversation

`Conversation` is the runtime shell around an agent. It owns the workspace, event stream, persistence, callbacks, hooks, visualization, and the execution loop used to send messages and run the agent.

Use this section when you need to start, resume, pause, observe, or clean up an agent run.

Implementation details: [`sdk_reference_details/04_conversation.md`](sdk_reference_details/04_conversation.md)

---

## 5. AgentContext

`AgentContext` carries reusable prompt and runtime context such as skills, prompt suffixes, secrets, and time context. It is the main place to attach durable context to an agent.

Use this section when you want an agent to carry the same instructions or supporting context across turns.

Implementation details: [`sdk_reference_details/05_agent_context.md`](sdk_reference_details/05_agent_context.md)

---

## 6. Skill

`Skill` packages instructions, examples, or configuration that can be injected unconditionally or triggered only when relevant. Skills are the SDK's built-in mechanism for progressive disclosure.

Use this section when you want reusable task guidance without hard-coding everything into the base system prompt.

Implementation details: [`sdk_reference_details/06_skill.md`](sdk_reference_details/06_skill.md)

---

## 7. Custom Tool Definition Pattern

The custom tool definition pattern is the SDK's extension path for authoring new tools. It defines how actions, observations, executors, tool definitions, and registration fit together.

Use this section when built-in tools or MCP servers are not enough and you need a first-class custom tool with its own schema and observation handling.

Implementation details: [`sdk_reference_details/07_custom_tool_definition_pattern.md`](sdk_reference_details/07_custom_tool_definition_pattern.md)

---

## 8. Condensers

Condensers control how long-running conversations retain or compress context. They help manage event growth so an agent can continue operating when history becomes large.

Use this section when you need summarization, trimming, or a pipeline of context-reduction strategies.

Implementation details: [`sdk_reference_details/08_condensers.md`](sdk_reference_details/08_condensers.md)

---

## 9. Hooks

Hooks let you run shell-based logic at specific lifecycle points such as before tool execution, after tool execution, session start, or session end.

Use this section when you need policy enforcement, auditing, logging, or runtime interception around agent behavior.

Implementation details: [`sdk_reference_details/09_hooks.md`](sdk_reference_details/09_hooks.md)

---

## 10. Built-in Tools

The SDK ships with a standard tool catalog for common coding and workspace tasks, plus default completion and reasoning tools that are included unless explicitly changed.

Use this section when you need to know what is available out of the box or when you want to start from preset agent factories.

Implementation details: [`sdk_reference_details/10_built_in_tools.md`](sdk_reference_details/10_built_in_tools.md)

---

## 11. MCP Integration

MCP integration lets an agent expose tools from external MCP servers without writing a custom OpenHands tool implementation for each one.

Use this section when you want to add external tool servers or selectively filter the MCP tool surface that the agent can see.

Implementation details: [`sdk_reference_details/11_mcp_integration.md`](sdk_reference_details/11_mcp_integration.md)

---

## 12. Agent Delegation

Agent delegation lets one agent hand work to specialized sub-agents. It is the SDK mechanism for splitting tasks across separate tool sets, prompts, and context windows.

Use this section when you need specialist workers or hierarchical execution.

Implementation details: [`sdk_reference_details/12_agent_delegation.md`](sdk_reference_details/12_agent_delegation.md)

---

## 13. Critic (Experimental)

The critic API adds a runtime evaluation loop that can score or reject agent behavior and optionally trigger iterative refinement.

Use this section when you need external evaluation during execution rather than after the fact.

Implementation details: [`sdk_reference_details/13_critic.md`](sdk_reference_details/13_critic.md)

---

## 14. Events and Callbacks

Events and callbacks are the SDK's observability layer. They let callers inspect messages, actions, observations, errors, and other execution artifacts as they happen.

Use this section when you need logging, analytics, custom UI updates, or post-processing of the agent trajectory.

Implementation details: [`sdk_reference_details/14_events_and_callbacks.md`](sdk_reference_details/14_events_and_callbacks.md)

---

## 15. LLM Registry

`LLMRegistry` stores model clients by usage ID so they can be tracked and reused consistently across different parts of an application.

Use this section when you want stable access to multiple model roles such as main-agent, condenser, or low-cost helper models.

Implementation details: [`sdk_reference_details/15_llm_registry.md`](sdk_reference_details/15_llm_registry.md)

---

## 16. Common Patterns

This section collects representative compositions of the SDK, such as minimal agents, condensed conversations, multi-phase workflows, callback collection, and custom prompts.

Use this section when you want an end-to-end starting point rather than an isolated API surface.

Implementation details: [`sdk_reference_details/16_common_patterns.md`](sdk_reference_details/16_common_patterns.md)

---

## 17. Retriever Tool

`BM25ToolSet` is an in-memory BM25 retrieval toolset that gives an agent two coordinated tools: one to index a document corpus at runtime and one to query it. It requires no external dependencies and is designed to be extended with custom retrieval backends.

Use this section when you need an agent to search over a large document corpus rather than reading it all into context.

Implementation details: [`sdk_reference_details/17_retriever_tool.md`](sdk_reference_details/17_retriever_tool.md)
