"""Pytest configuration."""

import pytest


@pytest.fixture(autouse=True)
def _provider_test_credentials(monkeypatch):
    """Allow mount/shape tests without requiring real Anthropic credentials."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-api-key")
