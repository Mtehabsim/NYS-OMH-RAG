"""ETL pipeline: downloads PDFs, cleans them, chunks text, and builds the ChromaDB vector store."""

import os
import shutil

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich.console import Console

from src import constants
from src.document_fetcher import DocumentFetcher
from src.models import DocumentChunk
from src.utils import init_environment

console = Console()


def parse_clean_pdf(pdf_path: str, source_title: str) -> list[DocumentChunk]:
    """Extracts body text from a PDF, cropping header/footer margins.

    Looks up optimal crop margins per PDF to ensure headers are removed
    without clipping valid body text.
    """
    chunks: list[DocumentChunk] = []

    margins = constants.PDF_CROP_MARGINS.get(source_title.upper())
    top_ratio = margins["top"] if margins else constants.DEFAULT_CROP_MARGIN_TOP
    bottom_ratio = margins["bottom"] if margins else constants.DEFAULT_CROP_MARGIN_BOTTOM

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            width = page.width
            height = page.height
            bbox = (
                0,
                height * top_ratio,
                width,
                height * bottom_ratio,
            )
            cropped_page = page.crop(bbox)
            text = cropped_page.extract_text()
            if text:
                chunks.append(DocumentChunk(text=text, page=i + 1, source=source_title))

    return chunks


def main() -> None:
    """Runs the full ingestion pipeline."""
    init_environment(require_db=False)

    # Clear stale database to prevent duplicate embeddings on re-runs
    if os.path.exists(constants.DB_DIR):
        console.print(
            "[yellow]Existing database found — rebuilding from scratch...[/yellow]"
        )
        shutil.rmtree(constants.DB_DIR)

    console.print("[bold green]Starting ingestion...[/bold green]")

    fetcher = DocumentFetcher(urls=constants.POLICY_URLS, target_dir=constants.DATA_DIR)
    pdf_paths = fetcher.fetch()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=constants.CHUNK_SIZE,
        chunk_overlap=constants.CHUNK_OVERLAP,
        separators=constants.CHUNK_SEPARATORS,
    )

    all_texts: list[str] = []
    all_metadatas: list[dict] = []

    for filepath, filename in pdf_paths:
        title = filename.replace(".pdf", "").upper()
        console.print(f"Processing [bold cyan]{title}[/bold cyan]...")

        page_chunks = parse_clean_pdf(filepath, title)
        for chunk in page_chunks:
            for segment in text_splitter.split_text(chunk.text):
                all_texts.append(segment)
                all_metadatas.append({"source": chunk.source, "page": str(chunk.page)})

    console.print(f"Total chunks created: [bold yellow]{len(all_texts)}[/bold yellow]")

    console.print("[yellow]Embedding and storing in ChromaDB...[/yellow]")
    embeddings = GoogleGenerativeAIEmbeddings(model=constants.EMBEDDING_MODEL)

    Chroma.from_texts(
        texts=all_texts,
        metadatas=all_metadatas,
        embedding=embeddings,
        persist_directory=constants.DB_DIR,
    )

    console.print(
        f"[bold green]Ingestion complete. Database saved to {constants.DB_DIR}/[/bold green]"
    )


if __name__ == "__main__":
    main()
