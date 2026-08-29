"""Retrieval and generation logic for the RAG pipeline."""

from __future__ import annotations

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

from src import constants
from src.models import QueryResult, SourceMetadata
from src.prompts import PromptType, get_system_prompt


class InferenceEngine:
    """Retrieves relevant chunks from the vector store and generates cited answers.

    Encapsulates the retrieval + generation chain so the CLI layer stays thin.
    """

    def __init__(self, vectorstore: Chroma, llm: ChatGoogleGenerativeAI) -> None:
        self.vectorstore = vectorstore
        self.llm = llm

        system_prompt = get_system_prompt(PromptType.CHAIN_OF_THOUGHT)
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{question}"),
            ]
        )
        self.chain = prompt_template | self.llm

    def generate_answer(self, query: str) -> QueryResult:
        """Retrieves top-K chunks and generates a cited answer."""
        results = self.vectorstore.similarity_search(query, k=constants.RETRIEVER_K)

        context_parts: list[str] = []
        sources: list[SourceMetadata] = []
        retrieved_chunks: list[str] = []

        for doc in results:
            source = doc.metadata.get("source", "UNKNOWN")
            page = doc.metadata.get("page", "?")
            meta = SourceMetadata(source=source, page=str(page))

            context_parts.append(
                f"--- Document: {source}, Page {page} ---\n{doc.page_content}"
            )
            sources.append(meta)
            retrieved_chunks.append(doc.page_content)

        context_str = "\n\n".join(context_parts)
        response = self.chain.invoke({"context": context_str, "question": query})

        return QueryResult(
            answer=response.content,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
        )
