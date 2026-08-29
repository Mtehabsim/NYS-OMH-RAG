"""Interactive CLI for querying OMH policies via RAG."""

import argparse

from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src import constants
from src.inference import InferenceEngine
from src.utils import init_environment

console = Console()


def main() -> None:
    """Runs the interactive query loop."""
    parser = argparse.ArgumentParser(description="OMH Policy RAG CLI")
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Display the retrieved document chunks before the answer.",
    )
    args = parser.parse_args()

    init_environment(require_db=True)

    console.print("[bold green]Loading vector database...[/bold green]")
    embeddings = GoogleGenerativeAIEmbeddings(model=constants.EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=constants.DB_DIR, embedding_function=embeddings
    )
    llm = ChatGoogleGenerativeAI(
        model=constants.LLM_MODEL, temperature=constants.LLM_TEMPERATURE
    )
    engine = InferenceEngine(vectorstore=vectorstore, llm=llm)

    console.print(
        Panel.fit(
            "OMH Policy RAG System\nType your question below, or 'exit' to quit.",
            title="Welcome",
            border_style="blue",
        )
    )

    while True:
        try:
            query = console.input("\n[bold cyan]Question:[/bold cyan] ")
            if query.lower().strip() in constants.EXIT_COMMANDS:
                break
            if not query.strip():
                continue

            with console.status(
                "[yellow]Searching policies and generating answer...[/yellow]"
            ):
                result = engine.generate_answer(query)

            if args.verbose:
                console.print("\n[bold]Retrieved Chunks:[/bold]")
                for i, (chunk, src) in enumerate(
                    zip(result.retrieved_chunks, result.sources)
                ):
                    console.print(
                        Panel(
                            chunk,
                            title=f"Chunk {i + 1}: {src.source}, Page {src.page}",
                            border_style="dim",
                        )
                    )

            console.print(
                Panel(Markdown(result.answer), title="Answer", border_style="green")
            )

            console.print("\n[dim]Sources Retrieved:[/dim]")
            for src in result.sources:
                console.print(f"[dim]- {src.source}, Page {src.page}[/dim]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


if __name__ == "__main__":
    main()
