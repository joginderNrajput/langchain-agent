"""Command-line interface.

Exposes three commands:
    ask     one-shot question → answer
    chat    interactive REPL with conversation memory
    ingest  (re)build the knowledge base vector store

Installed as the ``research-agent`` console script (see pyproject.toml).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agentic_research_agent.agents.service import ResearchAgent
from agentic_research_agent.config.settings import get_settings
from agentic_research_agent.core.exceptions import AgentError
from agentic_research_agent.tools.knowledge_base import KnowledgeBase

app = typer.Typer(
    name="research-agent",
    help="An enterprise-grade AI research assistant (LangChain + LangGraph).",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    thread_id: str = typer.Option("default", "--thread", "-t", help="Conversation id."),
) -> None:
    """Answer a single QUESTION and exit."""

    agent = _build_agent()
    try:
        with console.status("[bold cyan]Researching…", spinner="dots"):
            response = agent.ask(question, thread_id=thread_id)
    except AgentError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    console.print(Panel(Markdown(response.answer), title="Answer", border_style="green"))
    if response.tool_calls:
        used = ", ".join(sorted({tc.name for tc in response.tool_calls}))
        console.print(f"[dim]Tools used: {used}[/dim]")


@app.command()
def chat(
    question: str | None = typer.Argument(
        None, help="Optional first question to seed the session."
    ),
    thread_id: str = typer.Option("default", "--thread", "-t", help="Conversation id."),
) -> None:
    """Start an interactive chat session (type 'exit' or Ctrl-C to quit).

    Pass an optional QUESTION to ask immediately before the prompt opens, e.g.
    ``research-agent chat "Compare the top AI startups funded in 2024."``
    """

    agent = _build_agent()
    console.print(
        Panel(
            "Ask me anything. I can search the web, query the knowledge base, "
            "and do exact math.\nType [bold]exit[/bold] to quit.",
            title="Research Assistant",
            border_style="cyan",
        )
    )

    # If a question was supplied on the command line, answer it first.
    pending = question.strip() if question else None

    while True:
        if pending:
            user_input, pending = pending, None
        else:
            try:
                user_input = console.input("[bold blue]you ›[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                raise typer.Exit(0) from None

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", ":q"}:
            console.print("[dim]Goodbye.[/dim]")
            raise typer.Exit(0)

        try:
            with console.status("[bold cyan]Thinking…", spinner="dots"):
                response = agent.ask(user_input, thread_id=thread_id)
        except AgentError as exc:
            console.print(f"[bold red]Error:[/bold red] {exc}")
            continue
        console.print("[bold green]agent ›[/bold green] ", end="")
        console.print(Markdown(response.answer))


@app.command()
def ingest(
    force: bool = typer.Option(
        False, "--force", "-f", help="Rebuild the vector store from scratch."
    ),
) -> None:
    """Build (or rebuild) the knowledge base vector store from documents."""

    settings = get_settings()
    console.print(f"[cyan]Ingesting documents from[/cyan] {settings.knowledge_base_dir}")
    kb = KnowledgeBase(settings)
    with console.status("[bold cyan]Embedding & indexing…", spinner="dots"):
        kb.build(force=force)
    console.print(f"[green]✓ Vector store ready at[/green] {settings.vector_store_dir}")


def _build_agent() -> ResearchAgent:
    """Construct the agent, turning config errors into clean CLI messages."""

    try:
        return ResearchAgent()
    except AgentError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from exc


def main() -> None:
    """Console-script entry point."""

    app()


if __name__ == "__main__":
    main()
