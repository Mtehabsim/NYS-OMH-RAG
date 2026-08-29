"""Shared test fixtures and configuration."""

import pytest


@pytest.fixture(autouse=True)
def mock_openai_key(monkeypatch):
    """Ensures tests never require or leak a real API key."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-unit-tests")
