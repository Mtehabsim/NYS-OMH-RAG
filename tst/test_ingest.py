from unittest.mock import patch, MagicMock

from src.ingest import parse_clean_pdf
from src.models import DocumentChunk


@patch("src.ingest.pdfplumber.open")
def test_parse_clean_pdf_extracts_body(mock_pdfplumber_open):
    """Text should be extracted from the cropped body region."""
    mock_page = MagicMock()
    mock_page.width = 100
    mock_page.height = 100

    mock_cropped = MagicMock()
    mock_cropped.extract_text.return_value = "Test content paragraph."
    mock_page.crop.return_value = mock_cropped

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    chunks = parse_clean_pdf("dummy_path.pdf", "OM-500")

    assert len(chunks) == 1
    assert isinstance(chunks[0], DocumentChunk)
    assert chunks[0].text == "Test content paragraph."
    assert chunks[0].page == 1
    assert chunks[0].source == "OM-500"

    # Verify the crop box matches the new optimized OM-500 heuristic (10% top, no bottom crop)
    mock_page.crop.assert_called_once_with((0, 10.0, 100, 100.0))


@patch("src.ingest.pdfplumber.open")
def test_parse_clean_pdf_skips_empty_pages(mock_pdfplumber_open):
    """Pages with no extractable text after cropping should be skipped."""
    mock_page = MagicMock()
    mock_page.width = 100
    mock_page.height = 100

    mock_cropped = MagicMock()
    mock_cropped.extract_text.return_value = None
    mock_page.crop.return_value = mock_cropped

    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    chunks = parse_clean_pdf("dummy.pdf", "OM-500")

    assert len(chunks) == 0


@patch("src.ingest.pdfplumber.open")
def test_parse_clean_pdf_multi_page(mock_pdfplumber_open):
    """Multi-page PDFs should produce one chunk per page with correct page numbers."""
    pages = []
    for _ in range(3):
        mock_page = MagicMock()
        mock_page.width = 612
        mock_page.height = 792
        mock_cropped = MagicMock()
        mock_cropped.extract_text.return_value = "Some policy text."
        mock_page.crop.return_value = mock_cropped
        pages.append(mock_page)

    mock_pdf = MagicMock()
    mock_pdf.pages = pages
    mock_pdfplumber_open.return_value.__enter__.return_value = mock_pdf

    chunks = parse_clean_pdf("test.pdf", "OM-505")

    assert len(chunks) == 3
    assert [c.page for c in chunks] == [1, 2, 3]
    assert all(c.source == "OM-505" for c in chunks)
