"""Tests for task-6: RetryConfig uses Rust-aligned field names.

Verifies:
- RetryConfig constructed with initial_delay= (not min_delay=)
- RetryConfig constructed with jitter=bool (not jitter=float)
- No jitter bool/float compat code remains (no isinstance coercion)
- Config keys (min_retry_delay, max_retry_delay, retry_jitter, max_retries) unchanged
"""

from unittest.mock import MagicMock, patch

from amplifier_module_provider_anthropic import AnthropicProvider


def _make_provider_with_mock_retry_config(config: dict | None = None):
    """Create a provider while capturing the RetryConfig constructor call."""
    mock_retry_config_cls = MagicMock()
    mock_retry_config_instance = MagicMock()
    mock_retry_config_cls.return_value = mock_retry_config_instance

    with patch(
        "amplifier_module_provider_anthropic.RetryConfig", mock_retry_config_cls
    ):
        provider = AnthropicProvider(api_key="test-key", config=config or {})

    return provider, mock_retry_config_cls


class TestRetryConfigFieldNames:
    """RetryConfig must use Rust-aligned field names."""

    def test_uses_initial_delay_not_min_delay(self):
        """RetryConfig should use initial_delay=, not min_delay=."""
        _, mock_cls = _make_provider_with_mock_retry_config({"min_retry_delay": 2.5})
        call_kwargs = mock_cls.call_args[1]
        assert "initial_delay" in call_kwargs, (
            f"Expected 'initial_delay' kwarg, got: {list(call_kwargs.keys())}"
        )
        assert "min_delay" not in call_kwargs, (
            "'min_delay' should not be used — use 'initial_delay' per Rust alignment"
        )
        assert call_kwargs["initial_delay"] == 2.5

    def test_jitter_is_bool_not_float(self):
        """RetryConfig jitter= should be a bool, not a float."""
        _, mock_cls = _make_provider_with_mock_retry_config({"retry_jitter": True})
        call_kwargs = mock_cls.call_args[1]
        assert "jitter" in call_kwargs
        assert call_kwargs["jitter"] is True
        assert isinstance(call_kwargs["jitter"], bool)

    def test_jitter_false_passed_as_bool(self):
        """retry_jitter=False should pass jitter=False (bool), not 0.0 (float)."""
        _, mock_cls = _make_provider_with_mock_retry_config({"retry_jitter": False})
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["jitter"] is False
        assert isinstance(call_kwargs["jitter"], bool)

    def test_jitter_default_is_true(self):
        """When retry_jitter not in config, jitter defaults to True."""
        _, mock_cls = _make_provider_with_mock_retry_config({})
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["jitter"] is True

    def test_config_keys_unchanged(self):
        """Config keys (min_retry_delay, max_retry_delay, etc.) map to new field names."""
        _, mock_cls = _make_provider_with_mock_retry_config(
            {
                "min_retry_delay": 0.5,
                "max_retry_delay": 30.0,
                "retry_jitter": False,
                "max_retries": 10,
            }
        )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["initial_delay"] == 0.5
        assert call_kwargs["max_delay"] == 30.0
        assert call_kwargs["jitter"] is False
        assert call_kwargs["max_retries"] == 10


class TestNoJitterCompatCode:
    """The old jitter bool/float compat code must be removed."""

    def test_no_isinstance_jitter_coercion_in_source(self):
        """Source should not contain isinstance-based jitter coercion."""
        import inspect

        source = inspect.getsource(AnthropicProvider.__init__)
        assert "isinstance(jitter_val" not in source, (
            "Old jitter bool/float compat code (isinstance check) should be removed"
        )
        assert "jitter_val" not in source, (
            "Old jitter_val variable should be removed entirely"
        )
