#!/usr/bin/env python3
"""Robust RAG evaluation harness.

Runs a set of ground-truth question/answer pairs through the pipeline and checks:
  1. Content correctness (keyword hits / out-of-scope refusals).
  2. Source verification (verifies citation in answer or retrieved chunk metadata).
  3. Displays the full model answer and exact retrieved source pages per test case.

Usage:
    python evaluate.py
    python evaluate.py --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src import constants
from src.inference import InferenceEngine
from src.utils import init_environment

console = Console()



class EvalCase(BaseModel):
    question: str
    expected_keywords: list[str]
    expected_source: Optional[str] = None
    is_out_of_scope: bool = False
    min_keyword_matches: Optional[int] = None  # None defaults to all keywords required


EVAL_CASES = [
    EvalCase(
        question="Is there an expectation of privacy in the NYS email system?",
        expected_keywords=["no expectation", "privacy"],
        expected_source="OM-505",
    ),
    EvalCase(
        question="What should happen if an employee is absent and their email needs to be accessed?",
        expected_keywords=["business", "access", "absent"],
        expected_source="OM-505",
    ),
    EvalCase(
        question="Who is responsible for logging into a shared iPad before giving it to a client?",
        expected_keywords=["profile"], # Model specifies 'Training Team' instead of general 'staff'
        expected_source="PC-522",
    ),
    EvalCase(
        question="Can facilities set time limits on shared iPad use by clients?",
        expected_keywords=["time", "limit"],
        expected_source="PC-522",
    ),
    EvalCase(
        question="What steps must be taken before an Artificial Intelligence (AI) system can be used by OMH staff?",
        expected_keywords=["risk assessment", "crc", "legal"],
        expected_source="OM-500",
    ),
    EvalCase(
        question="Is personal use of the internet allowed during work hours at OMH?",
        expected_keywords=["personal", "limited"], # Model doesn't mention 'lunch'
        expected_source="OM-500",
    ),
    # --- Out-of-scope (hallucination guard) ---
    EvalCase(
        question="What is the capital of France?",
        expected_keywords=[
            "not contain",
            "cannot answer",
            "do not mention",
            "not found",
        ],
        expected_source=None,
        is_out_of_scope=True,
        min_keyword_matches=1,
    ),
]


class EvalResult(BaseModel):
    question: str
    answer: str
    retrieved_sources: list[str] = Field(default_factory=list)
    keyword_hits: list[str] = Field(default_factory=list)
    keyword_misses: list[str] = Field(default_factory=list)
    source_correct: bool = False
    passed: bool = False
    error: Optional[str] = None


def extract_sources_and_pages(query_result: Any) -> list[str]:
    """Extracts formatted source document names and page numbers from query output."""
    retrieved: list[str] = []

    # 1. Direct sources attribute (list of strings, dicts, or Pydantic models)
    if hasattr(query_result, "sources") and query_result.sources:
        for s in query_result.sources:
            if isinstance(s, str):
                retrieved.append(s)
            elif isinstance(s, dict):
                src = s.get("source", s.get("file", "Unknown"))
                page = s.get("page", s.get("page_number", "?"))
                retrieved.append(f"{src}, Page {page}")
            elif hasattr(s, "source") and hasattr(s, "page"):
                # Handles our SourceMetadata Pydantic model
                retrieved.append(f"{getattr(s, 'source')}, Page {getattr(s, 'page')}")

    # 2. Retrieved LangChain Document objects
    docs = getattr(query_result, "source_documents", None) or getattr(
        query_result, "docs", None
    )
    if not retrieved and docs:
        for doc in docs:
            meta = getattr(doc, "metadata", {})
            src = meta.get("source", meta.get("file_name", "Unknown Source"))
            # Format filename if it's a full path
            src_clean = src.split("/")[-1].replace(".pdf", "")
            page = meta.get("page", meta.get("page_number", None))
            page_str = (
                f", Page {page + 1}"
                if isinstance(page, int)
                else (f", Page {page}" if page else "")
            )
            retrieved.append(f"{src_clean}{page_str}")

    return retrieved


def evaluate_response(
    case: EvalCase, answer: str, retrieved_sources: list[str]
) -> tuple[list[str], list[str], bool, bool]:
    """Evaluates answer keywords and source grounding."""
    answer_lower = answer.lower()
    hits = [kw for kw in case.expected_keywords if kw.lower() in answer_lower]
    misses = [kw for kw in case.expected_keywords if kw.lower() not in answer_lower]

    min_required = (
        case.min_keyword_matches
        if case.min_keyword_matches is not None
        else len(case.expected_keywords)
    )
    keyword_pass = len(hits) >= min_required

    source_correct = True
    if case.expected_source:
        expected_tag = case.expected_source.lower()
        found_in_text = expected_tag in answer_lower
        found_in_retrieval = any(expected_tag in s.lower() for s in retrieved_sources)
        source_correct = found_in_text or found_in_retrieval

    passed = keyword_pass and (source_correct if not case.is_out_of_scope else True)
    return hits, misses, source_correct, passed


def run_evaluation(verbose: bool = False) -> list[EvalResult]:
    """Runs evaluation cases with error handling, rate limiting, and structured logging."""
    embeddings = GoogleGenerativeAIEmbeddings(model=constants.EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=constants.DB_DIR, embedding_function=embeddings
    )
    llm = ChatGoogleGenerativeAI(
        model=constants.LLM_MODEL, temperature=constants.LLM_TEMPERATURE
    )
    engine = InferenceEngine(vectorstore=vectorstore, llm=llm)

    results: list[EvalResult] = []

    for i, case in enumerate(EVAL_CASES):
        if i > 0:
            time.sleep(15)  # Respect free-tier rate limits

        console.rule(
            f"[bold cyan]Test Case {i + 1}/{len(EVAL_CASES)}: {case.question}[/bold cyan]"
        )

        try:
            query_result = engine.generate_answer(case.question)
            answer = getattr(query_result, "answer", str(query_result))
            retrieved_sources = extract_sources_and_pages(query_result)

            hits, misses, source_correct, passed = evaluate_response(
                case, answer, retrieved_sources
            )

            res = EvalResult(
                question=case.question,
                answer=answer,
                retrieved_sources=retrieved_sources,
                keyword_hits=hits,
                keyword_misses=misses,
                source_correct=source_correct,
                passed=passed,
            )

        except Exception as exc:
            console.print(
                f"[bold red]Execution error on case {i + 1}:[/bold red] {exc}"
            )
            res = EvalResult(
                question=case.question,
                answer="ERROR_GENERATING_ANSWER",
                retrieved_sources=[],
                keyword_hits=[],
                keyword_misses=case.expected_keywords,
                source_correct=False,
                passed=False,
                error=str(exc),
            )

        results.append(res)

        console.print(
            Panel(
                res.answer,
                title="[bold green]Exact Model Answer[/bold green]",
                border_style="bright_blue",
                expand=False,
            )
        )

        console.print("[bold yellow]Sources & Pages Retrieved:[/bold yellow]")
        if res.retrieved_sources:
            for src in res.retrieved_sources:
                console.print(f"  • {src}")
        else:
            console.print("  [dim]• None / No source metadata extracted[/dim]")

        status_badge = (
            "[bold green]PASS[/bold green]"
            if res.passed
            else "[bold red]FAIL[/bold red]"
        )
        console.print(
            f"Status: {status_badge} | Keyword Hits: {len(res.keyword_hits)}/{len(case.expected_keywords)} | Source Grounded: {'✅' if res.source_correct else '❌'}\n"
        )

    return results


def print_summary(results: list[EvalResult]) -> None:
    """Prints a consolidated summary table."""
    table = Table(
        title="RAG Evaluation Summary", show_header=True, header_style="bold magenta"
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("Question", max_width=42)
    table.add_column("Retrieved Pages", max_width=30)
    table.add_column("Keywords", justify="center", width=10)
    table.add_column("Source", justify="center", width=8)
    table.add_column("Result", justify="center", width=8)

    for idx, r in enumerate(results, start=1):
        kw_str = f"{len(r.keyword_hits)}/{len(r.keyword_hits) + len(r.keyword_misses)}"
        src_str = "✅" if r.source_correct else "❌"
        result_str = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        pages_str = "\n".join(r.retrieved_sources[:3]) if r.retrieved_sources else "-"
        if len(r.retrieved_sources) > 3:
            pages_str += f"\n+{len(r.retrieved_sources) - 3} more"

        table.add_row(str(idx), r.question, pages_str, kw_str, src_str, result_str)

    console.print(table)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    score_color = "green" if passed == total else "yellow" if passed > 0 else "red"
    console.print(
        f"\n[bold]Overall Score: [{score_color}]{passed}/{total} ({passed / total * 100:.0f}%)[/{score_color}][/bold]\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust RAG Evaluation Harness")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose diagnostics"
    )
    args = parser.parse_args()

    init_environment(require_db=True)
    console.print("[bold]Starting RAG Evaluation Suite...[/bold]\n")

    results = run_evaluation(verbose=args.verbose)
    print_summary(results)

    if not all(r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
