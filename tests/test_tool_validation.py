"""
Tests for Anthropic provider tool_use/tool_result validation and repair.

Verifies that:
1. Provider repairs incomplete tool sequences with synthetic results
2. Validation catches remaining inconsistencies
"""

import pytest
from amplifier_module_provider_anthropic import AnthropicProvider


def test_repair_creates_synthetic_user_message():
    """Provider repairs missing tool_result by creating synthetic user message."""
    provider = AnthropicProvider(api_key="test-key")

    # Anthropic format after _convert_messages: tool_use with no following message
    messages = [
        {"role": "user", "content": "Run ls"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_123", "name": "bash", "input": {"cmd": "ls"}},
            ],
        },
        # Missing tool_result message!
        {"role": "user", "content": "Thanks"},
    ]

    # Repair should replace user message content with synthetic results
    repaired, count, repairs = provider._repair_incomplete_tool_sequences(messages)
    assert count == 1
    assert len(repaired) == 3  # Same 3 messages, but content replaced
    assert repaired[2]["role"] == "user"
    assert isinstance(repaired[2]["content"], list)
    assert len(repaired[2]["content"]) == 1
    assert repaired[2]["content"][0]["type"] == "tool_result"
    assert repaired[2]["content"][0]["tool_use_id"] == "toolu_123"
    assert "SYSTEM ERROR" in repaired[2]["content"][0]["content"]


def test_validation_catches_orphaned_tool_use():
    """Provider validation catches tool_use without matching tool_result."""
    provider = AnthropicProvider(api_key="test-key")

    # Simulate broken state: tool_use but no tool_result
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll run a command"},
                {"type": "tool_use", "id": "toolu_orphan", "name": "bash", "input": {"cmd": "ls"}},
            ],
        },
        # Missing tool_result message!
    ]

    # After repair, this should no longer raise
    repaired, count, repairs = provider._repair_incomplete_tool_sequences(messages)
    assert count == 1
    assert len(repaired) == 2
    assert repaired[1]["role"] == "user"
    assert any("SYSTEM ERROR" in str(block.get("content", "")) for block in repaired[1]["content"])


def test_validation_catches_wrong_next_role():
    """Provider validation catches tool_use followed by non-user message."""
    provider = AnthropicProvider(api_key="test-key")

    # Simulate broken state: tool_use followed by assistant message instead of user
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_123", "name": "bash", "input": {}}],
        },
        {"role": "assistant", "content": "this shouldn't be here"},  # Should be user role!
    ]

    with pytest.raises(ValueError, match="expected 'user' with tool_results"):
        provider._validate_anthropic_tool_consistency(messages)


def test_repair_injects_into_existing_user_message():
    """Provider repairs missing tool_result by injecting into existing user message."""
    provider = AnthropicProvider(api_key="test-key")

    # Broken state: tool_use ID doesn't have matching tool_result
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_expected", "name": "bash", "input": {}},
                {"type": "tool_use", "id": "toolu_missing", "name": "read", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_expected", "content": "result 1"},
                # Missing tool_result for "toolu_missing"!
            ],
        },
    ]

    # Repair should inject synthetic result
    repaired, count, repairs = provider._repair_incomplete_tool_sequences(messages)
    assert count == 1
    assert len(repaired[1]["content"]) == 2
    assert repaired[1]["content"][1]["type"] == "tool_result"
    assert repaired[1]["content"][1]["tool_use_id"] == "toolu_missing"
    assert "SYSTEM ERROR" in repaired[1]["content"][1]["content"]


def test_validation_catches_orphaned_tool_result():
    """Provider validation catches orphaned tool_result (extra result not in tool_use)."""
    provider = AnthropicProvider(api_key="test-key")

    # Simulate broken state: tool_result ID that wasn't in the preceding tool_use
    # This happens when compaction or retry leaves stale results
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_correct", "name": "bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_correct", "content": "correct result"},
                {"type": "tool_result", "tool_use_id": "toolu_stale", "content": "orphaned from earlier"},
            ],
        },
    ]

    with pytest.raises(ValueError, match="without matching tool_use"):
        provider._validate_anthropic_tool_consistency(messages)


def test_validation_catches_extra_tool_result():
    """Provider validation catches tool_result with ID not in tool_use."""
    provider = AnthropicProvider(api_key="test-key")

    # Simulate broken state: Extra tool_result that doesn't match any tool_use
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_actual", "name": "bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_actual", "content": "correct result"},
                {"type": "tool_result", "tool_use_id": "toolu_extra", "content": "orphaned result"},
            ],
        },
    ]

    with pytest.raises(ValueError, match="without matching tool_use"):
        provider._validate_anthropic_tool_consistency(messages)


def test_validation_passes_correct_pairs():
    """Provider validation passes when tool pairs are correct."""
    provider = AnthropicProvider(api_key="test-key")

    # Correct state: tool_use followed by matching tool_result
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_123", "name": "bash", "input": {"cmd": "ls"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_123", "content": "file1.txt\nfile2.txt"}],
        },
    ]

    # Should not raise
    provider._validate_anthropic_tool_consistency(messages)


def test_validation_passes_multiple_tool_pairs():
    """Provider validation passes when multiple tool pairs are all correct."""
    provider = AnthropicProvider(api_key="test-key")

    # Correct state: Multiple tool_use blocks with matching tool_results
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {}},
                {"type": "tool_use", "id": "toolu_2", "name": "read", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "bash result"},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": "file content"},
            ],
        },
    ]

    # Should not raise
    provider._validate_anthropic_tool_consistency(messages)


def test_validation_passes_conversation_without_tools():
    """Provider validation passes for normal conversation without tools."""
    provider = AnthropicProvider(api_key="test-key")

    # Normal conversation, no tools
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
    ]

    # Should not raise
    provider._validate_anthropic_tool_consistency(messages)
