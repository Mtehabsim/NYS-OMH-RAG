"""Startup initialization and validation helpers."""

import os
import sys

from dotenv import load_dotenv
from rich.console import Console

from src import constants

console = Console()


def init_environment(require_db: bool = False) -> None:
    """Loads environment variables and validates required configuration."""
    load_dotenv()

    # Chroma reads telemetry config directly from os.environ, disable it for privacy
    os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "None")

    if not os.getenv("GOOGLE_API_KEY"):
        console.print(
            "[bold red]Error: GOOGLE_API_KEY not set. Please configure it in your .env file.[/bold red]"
        )
        sys.exit(1)

    if require_db and not os.path.exists(constants.DB_DIR):
        console.print(
            "[bold red]Error: Vector database not found. Run 'python -m src.ingest' first.[/bold red]"
        )
        sys.exit(1)
