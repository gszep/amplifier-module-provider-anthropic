"""Pytest configuration."""

import pytest


@pytest.fixture(autouse=True)
def _provider_test_credentials(monkeypatch):
    """Allow shape tests without credentials or live model-catalog requests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-api-key")
    from amplifier_module_provider_anthropic import AnthropicProvider

    async def static_models(self):
        return []

    monkeypatch.setattr(AnthropicProvider, "list_models", static_models)
