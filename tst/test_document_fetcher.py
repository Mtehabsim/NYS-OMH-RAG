import pytest
from unittest.mock import patch, MagicMock

from src.document_fetcher import DocumentFetcher


@patch("src.document_fetcher.requests.get")
@patch("src.document_fetcher.os.path.exists")
@patch("builtins.open", new_callable=MagicMock)
def test_fetcher_downloads_missing_files(mock_open, mock_exists, mock_get):
    """When files don't exist locally, they should be downloaded."""
    mock_exists.return_value = False

    mock_response = MagicMock()
    mock_response.content = b"fake pdf content"
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    fetcher = DocumentFetcher(
        urls=["http://example.com/a.pdf", "http://example.com/b.pdf"],
        target_dir="fake_dir",
    )
    paths = fetcher.fetch()

    assert len(paths) == 2
    assert mock_get.call_count == 2
    mock_response.raise_for_status.assert_called()

    filenames = sorted(p[1] for p in paths)
    assert filenames == ["a.pdf", "b.pdf"]


@patch("src.document_fetcher.requests.get")
@patch("src.document_fetcher.os.path.exists")
def test_fetcher_skips_existing_files(mock_exists, mock_get):
    """When files already exist locally, no HTTP requests should be made."""
    mock_exists.return_value = True

    fetcher = DocumentFetcher(urls=["http://example.com/a.pdf"], target_dir="fake_dir")
    paths = fetcher.fetch()

    assert len(paths) == 1
    assert mock_get.call_count == 0


@patch("src.document_fetcher.requests.get")
@patch("src.document_fetcher.os.path.exists")
def test_fetcher_raises_on_http_error(mock_exists, mock_get):
    """HTTP errors should propagate instead of silently writing bad data."""
    mock_exists.return_value = False

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("404 Not Found")
    mock_get.return_value = mock_response

    fetcher = DocumentFetcher(
        urls=["http://example.com/bad.pdf"], target_dir="fake_dir"
    )

    with pytest.raises(Exception, match="404"):
        fetcher.fetch()
