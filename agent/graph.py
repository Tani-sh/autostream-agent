"""
LangGraph StateGraph for the AutoStream Conversational AI Agent.

Graph topology:
    START
      │
      ▼
  classify_intent  ← runs on EVERY turn
      │
      ├─ "greeting"    ──────────────────────────────────► generate_response
      │
      ├─ "inquiry"     ───────────────────────────────────► retrieve_knowledge
      │                                                          │
      │                                                          ▼
      │                                                   generate_response
      │
      └─ "high_intent" ───────────────────────────────────► collect_lead_info
                                                                 │
                                                    ┌────────────┴────────────┐
                                              all fields?               missing fields?
                                                    │                         │
                                                    ▼                         ▼
                                             capture_lead            generate_response
                                                    │                (asks for next field)
                                                    ▼
                                             generate_response
                                             (warm confirmation)
                                                    │
                                                   END

Key design decisions:
- `classify_intent` is the entry point for every turn, enabling continuous
  re-evaluation and mid-flow re-routing (e.g., inquiry interrupt during lead collection).
- `route_after_collect` checks the `awaiting_field` sentinel: if None, all fields
  are present and we can safely fire `capture_lead`.
- `MemorySaver` checkpointer persists the full AgentState across turns, scoped
  by `thread_id` — enabling genuine multi-turn memory (5–6+ turns).
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    classify_intent_node,
    retrieve_knowledge_node,
    collect_lead_info_node,
    capture_lead_node,
    generate_response_node,
)


# ---------------------------------------------------------------------------
# Routing functions (conditional edges)
# ---------------------------------------------------------------------------

def route_after_intent(state: AgentState) -> str:
    """
    Routes based on the classified intent.

    Special case: if the lead has already been captured, skip all lead
    collection logic and go straight to generating a response.
    """
    if state.get("lead_captured", False):
        return "generate_response"

    intent = state.get("intent", "inquiry")

    if intent == "greeting":
        return "generate_response"
    elif intent == "inquiry":
        return "retrieve_knowledge"
    elif intent == "high_intent":
        return "collect_lead_info"

    return "retrieve_knowledge"  # safe default


def route_after_collect(state: AgentState) -> str:
    """
    Routes after lead field extraction.

    If `awaiting_field` is None, all three fields are present → fire capture.
    Otherwise, route to generate_response to ask for the missing field.
    """
    if state.get("awaiting_field") is None:
        return "capture_lead"
    return "generate_response"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Builds and compiles the AutoStream LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    # ------ Nodes ------
    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("retrieve_knowledge", retrieve_knowledge_node)
    builder.add_node("collect_lead_info", collect_lead_info_node)
    builder.add_node("capture_lead", capture_lead_node)
    builder.add_node("generate_response", generate_response_node)

    # ------ Entry point: every turn starts at intent classification ------
    builder.add_edge(START, "classify_intent")

    # ------ Route based on intent ------
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "generate_response": "generate_response",
            "retrieve_knowledge": "retrieve_knowledge",
            "collect_lead_info": "collect_lead_info",
        },
    )

    # ------ After RAG retrieval → always generate ------
    builder.add_edge("retrieve_knowledge", "generate_response")

    # ------ After lead collection → capture or ask ------
    builder.add_conditional_edges(
        "collect_lead_info",
        route_after_collect,
        {
            "capture_lead": "capture_lead",
            "generate_response": "generate_response",
        },
    )

    # ------ After capture → generate confirmation ------
    builder.add_edge("capture_lead", "generate_response")

    # ------ Response is always the terminal node ------
    builder.add_edge("generate_response", END)

    # ------ Compile with MemorySaver for multi-turn persistence ------
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Singleton graph instance (imported by main.py)
# ---------------------------------------------------------------------------
graph = build_graph()
