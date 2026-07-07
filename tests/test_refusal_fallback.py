"""Tests for the refusal-fallback-to-opus feature.

When a model returns finish_reason="refusal", complete() retries exactly once
against a configured fallback model (default: claude-opus-4-8), with
thinking/redacted_thinking blocks stripped from assistant messages in the
retried request. Non-refusal responses pass through untouched, and the
fallback is skipped entirely when disabled or when it would route onto a
model in the same family that just refused (loop guard).

Covers:
  (a) Refusal triggers exactly one fallback call to the configured model;
      the fallback response is returned.
  (b) Non-refusal responses are returned untouched; no fallback call made.
  (c) _refusal_fallback_target returns None when disabled via config.
  (d) _refusal_fallback_target returns None when the configured target
      resolves to the same family as the refusing model (loop guard).
  (e) _strip_thinking_blocks does not mutate the original request, only
      removes thinking/redacted_thinking blocks from assistant messages,
      and leaves everything else untouched.
"""

import asyncio
from typing import cast
from unittest.mock import AsyncMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import (
    ChatRequest,
    ChatResponse,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
)

from amplifier_module_provider_anthropic import AnthropicProvider
from tests._helpers import FakeCoordinator


def _make_provider(default_model: str, **config_overrides) -> AnthropicProvider:
    provider = AnthropicProvider(
        api_key="test-key",
        config={
            "default_model": default_model,
            "max_retries": 0,
            **config_overrides,
        },
    )
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _request_with_refused_turn() -> ChatRequest:
    """A conversation whose last assistant turn carries thinking blocks --
    representative of what a real refusal-then-retry conversation looks like.
    """
    return ChatRequest(
        messages=[
            Message(role="user", content="Hello"),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="internal reasoning", signature="sig"),
                    RedactedThinkingBlock(data="opaque"),
                    TextBlock(text="partial reply"),
                ],
            ),
            Message(role="user", content="Please continue"),
        ]
    )


def _response(finish_reason: str, text: str = "ok") -> ChatResponse:
    return ChatResponse(
        content=[TextBlock(text=text)],
        finish_reason=finish_reason,
    )


def _assistant_content(message: Message) -> list:
    assert isinstance(message.content, list)
    return message.content


# ---------------------------------------------------------------------------
# (a) Refusal triggers exactly one fallback call; fallback response returned
# ---------------------------------------------------------------------------
def test_refusal_triggers_single_fallback_call_with_thinking_stripped():
    provider = _make_provider("claude-fable-5")
    request = _request_with_refused_turn()

    refusal_response = _response("refusal", text="")
    fallback_response = _response("end_turn", text="fallback answer")
    provider._complete_chat_request = AsyncMock(
        side_effect=[refusal_response, fallback_response]
    )

    result = asyncio.run(provider.complete(request))

    assert result is fallback_response
    assert provider._complete_chat_request.await_count == 2

    _, second_call = provider._complete_chat_request.await_args_list

    # Second (fallback) call goes to the configured fallback model.
    assert second_call.kwargs["model"] == "claude-opus-4-8"

    # The fallback request has thinking/redacted_thinking stripped from the
    # assistant message, other content untouched.
    fallback_request = second_call.args[0]
    assistant_msg = next(m for m in fallback_request.messages if m.role == "assistant")
    block_types = [b.type for b in _assistant_content(assistant_msg)]
    assert "thinking" not in block_types
    assert "redacted_thinking" not in block_types
    assert block_types == ["text"]

    # The original request object passed to complete() was not mutated.
    original_assistant_msg = next(m for m in request.messages if m.role == "assistant")
    original_block_types = [b.type for b in _assistant_content(original_assistant_msg)]
    assert "thinking" in original_block_types
    assert "redacted_thinking" in original_block_types


# ---------------------------------------------------------------------------
# (b) Non-refusal responses returned untouched -- no fallback call
# ---------------------------------------------------------------------------
def test_non_refusal_response_returned_untouched_no_fallback_call():
    provider = _make_provider("claude-fable-5")
    request = _request_with_refused_turn()

    normal_response = _response("end_turn", text="normal answer")
    provider._complete_chat_request = AsyncMock(return_value=normal_response)

    result = asyncio.run(provider.complete(request))

    assert result is normal_response
    assert provider._complete_chat_request.await_count == 1


# ---------------------------------------------------------------------------
# (c) _refusal_fallback_target returns None when disabled
# ---------------------------------------------------------------------------
def test_refusal_fallback_target_none_when_disabled():
    provider = _make_provider("claude-fable-5", refusal_fallback_enabled=False)
    assert provider._refusal_fallback_target("claude-fable-5") is None


# ---------------------------------------------------------------------------
# (d) _refusal_fallback_target returns None on same-family loop guard
# ---------------------------------------------------------------------------
def test_refusal_fallback_target_none_when_same_family():
    provider = _make_provider(
        "claude-opus-4-5", refusal_fallback_model="claude-opus-4-1"
    )
    assert provider._refusal_fallback_target("claude-opus-4-5") is None


# ---------------------------------------------------------------------------
# (e) _strip_thinking_blocks: no mutation, only thinking/redacted_thinking
#     removed from assistant messages, everything else untouched
# ---------------------------------------------------------------------------
def test_strip_thinking_blocks_does_not_mutate_and_only_strips_assistant_thinking():
    request = _request_with_refused_turn()
    original_assistant_content_ids = [
        id(block)
        for msg in request.messages
        if msg.role == "assistant"
        for block in _assistant_content(msg)
    ]

    stripped = AnthropicProvider._strip_thinking_blocks(request)

    # Original untouched.
    original_assistant_msg = next(m for m in request.messages if m.role == "assistant")
    assert [b.type for b in _assistant_content(original_assistant_msg)] == [
        "thinking",
        "redacted_thinking",
        "text",
    ]

    # Stripped copy has thinking/redacted_thinking removed, text preserved.
    stripped_assistant_msg = next(m for m in stripped.messages if m.role == "assistant")
    stripped_content = _assistant_content(stripped_assistant_msg)
    assert [b.type for b in stripped_content] == ["text"]
    assert cast(TextBlock, stripped_content[0]).text == "partial reply"

    # Non-assistant messages (string content) pass through untouched.
    stripped_user_msgs = [m for m in stripped.messages if m.role == "user"]
    original_user_msgs = [m for m in request.messages if m.role == "user"]
    assert [m.content for m in stripped_user_msgs] == [
        m.content for m in original_user_msgs
    ]

    # It's a deep copy -- the returned object is not the same instance/blocks.
    assert stripped is not request
    stripped_content_ids = [
        id(block)
        for msg in stripped.messages
        if msg.role == "assistant"
        for block in _assistant_content(msg)
    ]
    assert not set(stripped_content_ids) & set(original_assistant_content_ids)
