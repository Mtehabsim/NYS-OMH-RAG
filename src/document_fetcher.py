"""Downloads OMH policy PDFs to a local directory."""

import os

import requests
from rich.console import Console

console = Console()


class DocumentFetcher:
    """Downloads PDFs from a list of URLs to a local directory.

    Skips files that already exist locally. Validates HTTP responses
    and enforces a connection timeout to fail fast on network issues.
    """

    def __init__(self, urls: list[str], target_dir: str) -> None:
        self.urls = urls
        self.target_dir = target_dir

    def _download_single(self, url: str) -> tuple[str, str]:
        """Downloads a single PDF and returns (filepath, filename)."""
        filename = url.split("/")[-1]
        filepath = os.path.join(self.target_dir, filename)

        if os.path.exists(filepath):
            console.print(f"[dim]Found {filename} locally.[/dim]")
        else:
            console.print(f"[dim]Downloading {filename}...[/dim]")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)

        return filepath, filename

    def fetch(self) -> list[tuple[str, str]]:
        """Downloads all PDFs sequentially and returns their local paths."""
        os.makedirs(self.target_dir, exist_ok=True)
        return [self._download_single(url) for url in self.urls]
