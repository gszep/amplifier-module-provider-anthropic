"""
Tests for Anthropic provider tool_use/tool_result validation.

Verifies that the provider catches tool pair inconsistencies before
sending to the API, providing clear error messages.
"""

import pytest

from amplifier_module_provider_anthropic import AnthropicProvider


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

    with pytest.raises(ValueError, match="tool_use blocks but no following message"):
        provider._validate_anthropic_tool_consistency(messages)


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


def test_validation_catches_missing_tool_result_id():
    """Provider validation catches tool_use_id without matching tool_result."""
    provider = AnthropicProvider(api_key="test-key")

    # Simulate broken state: tool_use ID doesn't have matching tool_result
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

    with pytest.raises(ValueError, match="without matching tool_result"):
        provider._validate_anthropic_tool_consistency(messages)


def test_validation_catches_orphaned_tool_result():
    """Provider validation catches orphaned tool_result (extra result not in tool_use)."""
    provider = AnthropicProvider(api_key="test-key")

    # Simulate broken state: tool_result ID that wasn't in the preceding tool_use
    # This happens when compaction or retry leaves stale results
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_correct", "name": "bash", "input": {}}
            ],
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
