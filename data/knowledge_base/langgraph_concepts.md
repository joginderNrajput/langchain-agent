# LangGraph — Core Concepts

LangGraph is a library for building stateful, multi-step agent applications as
**graphs**. Where a simple chain is a fixed linear sequence, a LangGraph graph
supports branching, loops, and conditional routing — exactly what an agent's
reason-act loop needs.

## State

A graph is parameterized by a **state** schema (typically a `TypedDict`). Every
node receives the current state and returns a partial update. **Reducers**
control how updates merge: the built-in `add_messages` reducer *appends* new
messages to the history instead of replacing it, which is how an agent
accumulates context.

## Nodes and edges

- **Nodes** are functions that read state and return an update. In an agent,
  one node calls the LLM and another (`ToolNode`) executes tool calls.
- **Edges** connect nodes. **Conditional edges** route dynamically based on the
  current state — e.g. `tools_condition` sends flow to the tool node when the
  model requested a tool, otherwise to `END`.
- `START` and `END` are sentinels marking the graph's entry and exit.

## The agent loop in LangGraph

A minimal ReAct agent is: `START → agent → (tools? ) → tools → agent → … → END`.
The agent node proposes tool calls; the tool node runs them and feeds results
back; the loop repeats until the model answers without calling a tool.

## Checkpointers (memory & persistence)

A **checkpointer** snapshots graph state after each step, keyed by a
`thread_id`. This gives conversation memory across invocations and enables
human-in-the-loop pauses and time-travel debugging. `MemorySaver` keeps state
in memory; `SqliteSaver` and `PostgresSaver` persist it durably.

## Recursion limit

Because graphs can loop, LangGraph enforces a `recursion_limit` — the maximum
number of super-steps in a single run — to prevent an agent from looping
forever. Raise it for genuinely long tasks; if you hit it unexpectedly, it
usually signals the agent is stuck calling tools without converging.
