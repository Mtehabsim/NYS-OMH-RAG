"""Pydantic data models for the RAG pipeline."""

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A chunk of text extracted from a single PDF page before embedding."""

    text: str
    page: int
    source: str


class SourceMetadata(BaseModel):
    """Metadata for a single retrieved chunk."""

    source: str
    page: str


class QueryResult(BaseModel):
    """The answer and supporting evidence returned by the RAG chain."""

    answer: str
    sources: list[SourceMetadata] = Field(default_factory=list)
    retrieved_chunks: list[str] = Field(default_factory=list)
