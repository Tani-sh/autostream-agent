"""
LangGraph AgentState definition.

All fields are persisted across conversation turns via MemorySaver.
The `messages` field uses the `add_messages` reducer so new messages
are appended rather than overwriting the history.
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    # Full conversation history — grows across turns
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent classified on EVERY turn (re-evaluated continuously)
    # Values: "greeting" | "inquiry" | "high_intent"
    intent: str

    # RAG-retrieved context passages for the current turn
    retrieved_context: str

    # Partial or complete lead fields — persisted across turns
    lead_info: dict  # keys: "name", "email", "platform"

    # Which lead field we're currently waiting for (None when collection is complete)
    awaiting_field: Optional[str]

    # Set to True after mock_lead_capture fires — prevents double-firing
    lead_captured: bool
