"""
AutoStream Conversational AI Agent — CLI Entry Point

Rich terminal UI for demonstrating the full end-to-end agentic workflow:
  - RAG-powered product Q&A
  - Continuous intent classification
  - Structured lead capture

Usage:
    python main.py

Prerequisites:
    1. pip install -r requirements.txt
    2. Create a .env file: GOOGLE_API_KEY=your_key_here
"""

import os
import sys
from typing import Optional

from dotenv import load_dotenv

# Load .env before importing anything that needs the API key
load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    print(
        "\n❌  GOOGLE_API_KEY not set.\n"
        "   Create a .env file in the project root with:\n"
        "   GOOGLE_API_KEY=your_key_here\n"
        "   Get a free key at: https://aistudio.google.com/app/apikey\n"
    )
    sys.exit(1)

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.table import Table
from rich.rule import Rule
from rich import box
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import graph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
console = Console()

THREAD_ID = "autostream-demo-001"
CONFIG = {"configurable": {"thread_id": THREAD_ID}}

INTENT_STYLE = {
    "greeting":    ("💬 Casual Greeting", "cyan"),
    "inquiry":     ("🔍 Product Inquiry",  "yellow"),
    "high_intent": ("🚀 High Intent Lead", "green bold"),
}


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def print_banner() -> None:
    console.print()
    console.print(
        Panel.fit(
            "[bold white]AutoStream[/bold white] [dim]Sales Agent[/dim]  [blue]v1.0[/blue]\n"
            "[dim]Powered by LangGraph · Gemini 1.5 Flash · FAISS RAG[/dim]",
            border_style="blue",
            padding=(1, 4),
        )
    )
    console.print(
        "  Type [bold green]quit[/bold green] or [bold green]exit[/bold green] "
        "to end the session.  Type [bold green]reset[/bold green] to start over.\n"
    )


def print_agent_response(response: str) -> None:
    console.print(
        Panel(
            f"[white]{response}[/white]",
            title="[bold blue]🤖  Alex — AutoStream[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )


def print_state_summary(state: dict) -> None:
    """Render a compact diagnostic panel showing the current agent state."""
    intent = state.get("intent", "—")
    lead_info: dict = state.get("lead_info") or {}
    lead_captured: bool = state.get("lead_captured", False)
    awaiting: Optional[str] = state.get("awaiting_field")

    label, style = INTENT_STYLE.get(intent, (intent, "white"))

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1), show_edge=False)
    table.add_column("Key", style="dim", width=14)
    table.add_column("Value")

    table.add_row("Intent", f"[{style}]{label}[/{style}]")

    if lead_captured:
        table.add_row("Lead Status", "[bold green]✅ Captured — CRM updated[/bold green]")
    else:
        field_display = []
        for field in ("name", "email", "platform"):
            val = lead_info.get(field)
            if val:
                field_display.append(f"[green]{field}={val}[/green]")
            elif field == awaiting:
                field_display.append(f"[yellow]{field}=?[/yellow]")
            else:
                field_display.append(f"[dim]{field}=—[/dim]")
        table.add_row("Lead Fields", "  ".join(field_display))

    console.print(
        Panel(table, title="[dim]Agent State[/dim]", border_style="dim", padding=(0, 1))
    )
    console.print()


# ---------------------------------------------------------------------------
# Core Chat Function
# ---------------------------------------------------------------------------

def chat(user_input: str) -> tuple[str, dict]:
    """
    Invoke the LangGraph agent with a user message.

    Returns:
        (agent_response_text, final_state_dict)
    """
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
    )

    # Extract the last AI message
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    response_text = ai_messages[-1].content if ai_messages else "(no response)"

    return response_text, result


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def main() -> None:
    print_banner()

    # Pre-warm RAG pipeline (builds FAISS index on first run)
    console.print("[dim]  Initializing RAG pipeline …[/dim]")
    from rag.embedder import load_index
    load_index(verbose=False)
    console.print("[green]  ✅ Knowledge base ready.[/green]\n")

    console.print(Rule("[dim]Session started[/dim]", style="dim"))
    console.print()

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session ended. Goodbye! 👋[/dim]")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("quit", "exit", "bye"):
            console.print("[dim]Thanks for chatting! Goodbye! 👋[/dim]\n")
            break

        if user_input.lower() == "reset":
            global THREAD_ID, CONFIG
            import time
            THREAD_ID = f"autostream-{int(time.time())}"
            CONFIG = {"configurable": {"thread_id": THREAD_ID}}
            console.print("[yellow]  🔄 Session reset. Starting fresh conversation.[/yellow]\n")
            continue

        # Invoke agent
        with console.status("[dim]Alex is thinking …[/dim]", spinner="dots"):
            try:
                response, state = chat(user_input)
            except Exception as e:
                console.print(f"[red]  ⚠️  Error: {e}[/red]\n")
                continue

        print_agent_response(response)
        print_state_summary(state)


if __name__ == "__main__":
    main()
