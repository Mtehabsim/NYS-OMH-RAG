from unittest.mock import MagicMock

from src.inference import InferenceEngine


def test_generate_answer_with_valid_metadata():
    """Should retrieve chunks, format context, and return a cited answer."""
    mock_vectorstore = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "This is a test policy rule."
    mock_doc.metadata = {"source": "OM-500", "page": "2"}
    mock_vectorstore.similarity_search.return_value = [mock_doc]

    mock_llm = MagicMock()
    engine = InferenceEngine(vectorstore=mock_vectorstore, llm=mock_llm)

    # Override the chain for deterministic testing
    mock_chain = MagicMock()
    mock_response = MagicMock()
    mock_response.content = (
        "According to the test policy, this is a rule. [OM-500, Page 2]."
    )
    mock_chain.invoke.return_value = mock_response
    engine.chain = mock_chain

    result = engine.generate_answer("What is the test rule?")

    mock_vectorstore.similarity_search.assert_called_once_with(
        "What is the test rule?", k=5
    )

    assert len(result.sources) == 1
    assert result.sources[0].source == "OM-500"
    assert result.sources[0].page == "2"

    expected_context = "--- Document: OM-500, Page 2 ---\nThis is a test policy rule."
    mock_chain.invoke.assert_called_once_with(
        {
            "context": expected_context,
            "question": "What is the test rule?",
        }
    )

    assert (
        result.answer
        == "According to the test policy, this is a rule. [OM-500, Page 2]."
    )

    assert result.retrieved_chunks == ["This is a test policy rule."]


def test_generate_answer_with_missing_metadata():
    """Should gracefully handle chunks with missing or incomplete metadata."""
    mock_vectorstore = MagicMock()
    mock_doc = MagicMock()
    mock_doc.page_content = "Some content."
    mock_doc.metadata = {}  # No source or page
    mock_vectorstore.similarity_search.return_value = [mock_doc]

    mock_llm = MagicMock()
    engine = InferenceEngine(vectorstore=mock_vectorstore, llm=mock_llm)

    mock_chain = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Response"
    mock_chain.invoke.return_value = mock_response
    engine.chain = mock_chain

    result = engine.generate_answer("Query")

    assert result.sources[0].source == "UNKNOWN"
    assert result.sources[0].page == "?"


def test_generate_answer_multiple_chunks():
    """Should handle multiple retrieved chunks and preserve order."""
    mock_vectorstore = MagicMock()
    docs = []
    for i, src in enumerate([("OM-500", "1"), ("OM-505", "3"), ("PC-522", "2")]):
        doc = MagicMock()
        doc.page_content = f"Content from {src[0]}."
        doc.metadata = {"source": src[0], "page": src[1]}
        docs.append(doc)
    mock_vectorstore.similarity_search.return_value = docs

    mock_llm = MagicMock()
    engine = InferenceEngine(vectorstore=mock_vectorstore, llm=mock_llm)

    mock_chain = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Combined answer."
    mock_chain.invoke.return_value = mock_response
    engine.chain = mock_chain

    result = engine.generate_answer("Multi-source query")

    assert len(result.sources) == 3
    assert [s.source for s in result.sources] == ["OM-500", "OM-505", "PC-522"]
    assert len(result.retrieved_chunks) == 3
