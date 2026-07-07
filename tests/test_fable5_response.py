"""Tests for Fable 5 / Mythos-class API response handling.

Covers:
  (a) stop_reason='refusal' returns a ChatResponse without crashing
  (b) finish_reason propagates 'refusal'
  (c) stop_details absent (missing attr) does not crash
  (d) stop_details=None (normal completion) does not crash
  (e) usage.iterations absent does not crash
  (f) usage.iterations present does not crash
  (g) Unknown 'fallback' content block is skipped without crashing
"""

from unittest.mock import MagicMock

from amplifier_core import ModuleCoordinator

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator


def _make_provider() -> AnthropicProvider:
    coord: ModuleCoordinator = FakeCoordinator()  # type: ignore[assignment]
    provider = AnthropicProvider(api_key="test-key", config={"max_retries": 0})
    provider.coordinator = coord
    return provider


def _make_response(
    model: str = "claude-fable-5",
    stop_reason: str = "end_turn",
    content_blocks: list | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    has_stop_details: bool = False,
    stop_details=None,
    has_iterations: bool = False,
    iterations=None,
) -> MagicMock:
    """Build a minimal fake Anthropic API response for Fable 5 testing."""
    response = MagicMock()
    response.model = model
    response.stop_reason = stop_reason
    response.content = content_blocks if content_blocks is not None else []
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    response.usage.cache_read_input_tokens = 0
    response.usage.cache_creation_input_tokens = 0
    # speed is absent on Fable 5 (only on Opus 4.8 fast mode)
    del response.usage.speed
    # stop_details: present on all messages but None for non-refusal
    if has_stop_details:
        response.stop_details = stop_details
    else:
        # Simulate field missing entirely (older SDK / non-refusal)
        if hasattr(response, "stop_details"):
            del response.stop_details
    # usage.iterations: absent on non-fallback responses
    if has_iterations:
        response.usage.iterations = iterations
    else:
        if hasattr(response.usage, "iterations"):
            del response.usage.iterations
    return response


# ---------------------------------------------------------------------------
# (a) stop_reason='refusal' — must not crash
# ---------------------------------------------------------------------------
def test_refusal_stop_reason_does_not_crash():
    """stop_reason='refusal' must return a ChatResponse without crashing."""
    provider = _make_provider()
    response = _make_response(
        stop_reason="refusal",
        content_blocks=[],
        input_tokens=412,
        output_tokens=0,
    )
    result = provider._convert_to_chat_response(response)
    assert result is not None


# ---------------------------------------------------------------------------
# (b) finish_reason propagates 'refusal'
# ---------------------------------------------------------------------------
def test_refusal_finish_reason_propagated():
    """finish_reason in ChatResponse reflects 'refusal'."""
    provider = _make_provider()
    response = _make_response(stop_reason="refusal", content_blocks=[])
    result = provider._convert_to_chat_response(response)
    assert result.finish_reason == "refusal"


# ---------------------------------------------------------------------------
# (c) stop_details absent (field missing) — does not crash
# ---------------------------------------------------------------------------
def test_stop_details_absent_does_not_crash():
    """Response without stop_details attribute must not crash."""
    provider = _make_provider()
    response = _make_response(stop_reason="end_turn", has_stop_details=False)
    result = provider._convert_to_chat_response(response)
    assert result is not None


# ---------------------------------------------------------------------------
# (d) stop_details=None (field present, value None) — does not crash
# ---------------------------------------------------------------------------
def test_stop_details_none_does_not_crash():
    """stop_details=None on normal completion must not crash."""
    provider = _make_provider()
    response = _make_response(
        stop_reason="end_turn", has_stop_details=True, stop_details=None
    )
    result = provider._convert_to_chat_response(response)
    assert result is not None


# ---------------------------------------------------------------------------
# (e) usage.iterations absent — does not crash
# ---------------------------------------------------------------------------
def test_usage_iterations_absent_does_not_crash():
    """Response without usage.iterations (no fallback) must not crash."""
    provider = _make_provider()
    response = _make_response(has_iterations=False)
    result = provider._convert_to_chat_response(response)
    assert result is not None


# ---------------------------------------------------------------------------
# (f) usage.iterations present — does not crash
# ---------------------------------------------------------------------------
def test_usage_iterations_present_does_not_crash():
    """usage.iterations present (fallback ran) must not crash."""
    provider = _make_provider()
    iter1 = MagicMock(type="message")
    iter2 = MagicMock(type="fallback_message")
    response = _make_response(has_iterations=True, iterations=[iter1, iter2])
    result = provider._convert_to_chat_response(response)
    assert result is not None


# ---------------------------------------------------------------------------
# (g) Unknown 'fallback' content block — skipped without crashing
# ---------------------------------------------------------------------------
def test_fallback_content_block_skipped():
    """Unknown block type 'fallback' must be skipped, not crash."""
    provider = _make_provider()
    fallback_block = MagicMock()
    fallback_block.type = "fallback"

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello from fallback"

    response = _make_response(content_blocks=[fallback_block, text_block])
    result = provider._convert_to_chat_response(response)
    assert result is not None
