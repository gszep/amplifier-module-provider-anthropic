"""Tests for tool result repair."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message, ToolCallBlock

from amplifier_module_provider_claude import ClaudeProvider


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def test_tool_call_sequence_missing_tool_message_is_repaired():
    """Missing tool results should be repaired with synthetic results and emit event."""
    provider = ClaudeProvider(config={"use_streaming": False})
    provider._complete_chat_request = AsyncMock(return_value=MagicMock())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="do_something", input={"value": 1})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    provider._complete_chat_request.assert_awaited_once()

    assert all(
        event_name != "provider:validation_error"
        for event_name, _ in fake_coordinator.hooks.events
    )

    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "claude"
    assert repair_events[0][1]["repair_count"] == 1
    assert repair_events[0][1]["repairs"][0]["tool_name"] == "do_something"


def test_repaired_tool_ids_are_not_detected_again():
    """Repaired tool IDs should be tracked and not trigger re-detection."""
    provider = ClaudeProvider(config={"use_streaming": False})
    provider._complete_chat_request = AsyncMock(return_value=MagicMock())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    assert "call_abc123" in provider._repaired_tool_ids  # pyright: ignore[reportAttributeAccessIssue]
    repair_events_1 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_1) == 1

    fake_coordinator.hooks.events.clear()

    messages_2 = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_abc123", name="grep", input={"pattern": "test"})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request_2 = ChatRequest(messages=messages_2)

    asyncio.run(provider.complete(request_2))

    repair_events_2 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_2) == 0, "Should not re-detect already-repaired tool IDs"


def test_multiple_missing_tool_results_all_tracked():
    """Multiple missing tool results should all be tracked."""
    provider = ClaudeProvider(config={"use_streaming": False})
    provider._complete_chat_request = AsyncMock(return_value=MagicMock())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="grep", input={"pattern": "a"}),
                ToolCallBlock(id="call_2", name="grep", input={"pattern": "b"}),
                ToolCallBlock(id="call_3", name="grep", input={"pattern": "c"}),
            ],
        ),
        Message(role="user", content="No tool results"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    assert provider._repaired_tool_ids == {"call_1", "call_2", "call_3"}  # pyright: ignore[reportAttributeAccessIssue]

    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 3


def test_streaming_tool_call_sequence_missing_tool_message_is_repaired():
    """Missing tool results should be repaired with streaming API."""
    provider = ClaudeProvider()
    provider._complete_chat_request = AsyncMock(return_value=MagicMock())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="call_1", name="do_something", input={"value": 1})
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    # Should succeed (not raise validation error)
    provider._complete_chat_request.assert_awaited_once()

    assert all(
        event_name != "provider:validation_error"
        for event_name, _ in fake_coordinator.hooks.events
    )

    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["provider"] == "claude"
    assert repair_events[0][1]["repair_count"] == 1
    assert repair_events[0][1]["repairs"][0]["tool_name"] == "do_something"


def test_streaming_repaired_tool_ids_are_not_detected_again():
    """Repaired tool IDs should be tracked with streaming API."""
    provider = ClaudeProvider()
    provider._complete_chat_request = AsyncMock(return_value=MagicMock())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    # Create a request with missing tool result
    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(
                    id="call_stream_123", name="grep", input={"pattern": "test"}
                )
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request = ChatRequest(messages=messages)

    # First call - should detect and repair
    asyncio.run(provider.complete(request))

    assert "call_stream_123" in provider._repaired_tool_ids  # pyright: ignore[reportAttributeAccessIssue]
    repair_events_1 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_1) == 1

    # Clear events for second call
    fake_coordinator.hooks.events.clear()

    messages_2 = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(
                    id="call_stream_123", name="grep", input={"pattern": "test"}
                )
            ],
        ),
        Message(role="user", content="No tool result present"),
    ]
    request_2 = ChatRequest(messages=messages_2)

    asyncio.run(provider.complete(request_2))

    repair_events_2 = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events_2) == 0, "Should not re-detect already-repaired tool IDs"


def test_streaming_multiple_missing_tool_results_all_tracked():
    """Multiple missing tool results should all be tracked with streaming API."""
    provider = ClaudeProvider()
    provider._complete_chat_request = AsyncMock(return_value=MagicMock())
    fake_coordinator = FakeCoordinator()
    provider.coordinator = cast(ModuleCoordinator, fake_coordinator)

    messages = [
        Message(
            role="assistant",
            content=[
                ToolCallBlock(id="stream_1", name="grep", input={"pattern": "a"}),
                ToolCallBlock(id="stream_2", name="grep", input={"pattern": "b"}),
                ToolCallBlock(id="stream_3", name="grep", input={"pattern": "c"}),
            ],
        ),
        Message(role="user", content="No tool results"),
    ]
    request = ChatRequest(messages=messages)

    asyncio.run(provider.complete(request))

    assert provider._repaired_tool_ids == {"stream_1", "stream_2", "stream_3"}  # pyright: ignore[reportAttributeAccessIssue]

    repair_events = [
        e
        for e in fake_coordinator.hooks.events
        if e[0] == "provider:tool_sequence_repaired"
    ]
    assert len(repair_events) == 1
    assert repair_events[0][1]["repair_count"] == 3
