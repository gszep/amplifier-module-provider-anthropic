"""Tests for Anthropic subscription OAuth and native tool transport."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from amplifier_core.message_models import (
    ChatRequest,
    Message,
    ToolResultBlock,
    ToolSpec,
)
from anthropic.types import Message as AnthropicMessage

from amplifier_module_provider_anthropic import ClaudeProvider
from amplifier_module_provider_anthropic.auth import (
    AnthropicAuth,
    AnthropicAuthManager,
    OAUTH_BETAS,
    oauth_request_headers,
    read_credentials,
    write_credentials,
)


def test_oauth_headers_have_claude_code_identity(monkeypatch):
    monkeypatch.setenv("AMPLIFIER_CLAUDE_CODE_VERSION", "9.8.7")
    headers = oauth_request_headers()
    assert headers["x-app"] == "cli"
    assert headers["user-agent"] == "claude-cli/9.8.7 (external, cli)"
    assert set(headers["anthropic-beta"].split(",")) == set(OAUTH_BETAS)
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"


def test_credentials_are_written_atomically_with_private_permissions(tmp_path):
    path = tmp_path / "auth.json"
    credentials = {
        "type": "oauth",
        "access": "sk-ant-oat-access",
        "refresh": "refresh",
        "expires": 9999999999999,
    }
    write_credentials(path, credentials)
    assert read_credentials(path) == credentials
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_auth_manager_prefers_oauth_over_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "api-key")
    path = tmp_path / "auth.json"
    write_credentials(
        path,
        {
            "type": "oauth",
            "access": "sk-ant-oat-access",
            "refresh": "refresh",
            "expires": 9999999999999,
        },
    )
    auth = asyncio.run(AnthropicAuthManager(path=path).get_auth())
    assert auth == AnthropicAuth("sk-ant-oat-access", oauth=True)


def test_tool_results_remain_native_and_adjacent_users_are_merged():
    provider = ClaudeProvider()
    messages = [
        Message(
            role="user",
            content=[ToolResultBlock(tool_call_id="toolu_1", output="done")],
        ),
        Message(role="user", content="Continue"),
    ]
    converted = provider._convert_messages([message.model_dump() for message in messages])
    assert len(converted) == 1
    assert converted[0]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "done",
    }
    assert converted[0]["content"][1] == {"type": "text", "text": "Continue"}


def test_native_tools_are_sent_structurally(monkeypatch):
    provider = ClaudeProvider(config={"max_retries": 0})
    provider._auth.get_auth = AsyncMock(
        return_value=AnthropicAuth("sk-ant-oat-test", oauth=True)
    )

    response = AnthropicMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "Read",
                    "input": {"path": "README.md"},
                }
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    create = AsyncMock(return_value=response)
    provider._client_for_auth = AsyncMock(
        return_value=SimpleNamespace(messages=SimpleNamespace(create=create))
    )

    request = ChatRequest(
        messages=[Message(role="user", content="Read the README")],
        tools=[
            ToolSpec(
                name="read",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ],
    )
    result = asyncio.run(provider.complete(request))

    payload = create.await_args.kwargs
    assert payload["tools"][0]["name"] == "Read"
    assert payload["tools"][0]["input_schema"]["required"] == ["path"]
    assert payload["messages"][0]["role"] == "user"
    assert payload["messages"][0]["content"][0]["text"] == "Read the README"
    assert "<tools>" not in json.dumps(payload["messages"])
    assert "[tool]:" not in json.dumps(payload["messages"])
    assert payload["system"][0]["text"].startswith("You are Claude Code")
    assert result.tool_calls[0].name == "read"
    assert result.tool_calls[0].arguments == {"path": "README.md"}
