"""Tests for Anthropic subscription OAuth and native tool transport."""

import asyncio
import json
from io import BytesIO
import os
import subprocess
import sys
from urllib.error import HTTPError

from amplifier_core.message_models import ToolSpec
from anthropic.types import Message as AnthropicMessage
import pytest

from amplifier_module_provider_anthropic import AnthropicProvider
import amplifier_anthropic_oauth.auth as auth_module
from amplifier_anthropic_oauth.auth import (
    AnthropicAuth,
    AnthropicAuthError,
    AnthropicAuthManager,
    OAUTH_BETAS,
    oauth_request_headers,
    read_credentials,
    write_credentials,
)


def test_login_module_does_not_import_amplifier_core():
    code = """
import builtins
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'amplifier_core' or name.startswith('amplifier_core.'):
        raise AssertionError('standalone login imported amplifier_core')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import amplifier_anthropic_oauth.login
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_token_exchange_uses_claude_identity_headers(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("AMPLIFIER_CLAUDE_CODE_VERSION", "9.8.7")
    monkeypatch.setattr(auth_module, "urlopen", fake_urlopen)
    assert auth_module._post_json("https://example.test/token", {"code": "x"}) == {
        "ok": True
    }
    headers = dict(captured["request"].header_items())
    assert headers["User-agent"] == "claude-cli/9.8.7 (external, cli)"
    assert headers["Anthropic-beta"] == "oauth-2025-04-20"
    assert headers["X-app"] == "cli"


def test_token_exchange_error_includes_response_body(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            BytesIO(b'{"error":"invalid_request"}'),
        )

    monkeypatch.setattr(auth_module, "urlopen", fake_urlopen)
    with pytest.raises(AnthropicAuthError, match="invalid_request"):
        auth_module._post_json("https://example.test/token", {"code": "x"})


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


def test_api_key_config_is_optional_for_oauth_users():
    provider = AnthropicProvider(
        "sk-ant-oat-test",
        initial_auth=AnthropicAuth("sk-ant-oat-test", oauth=True),
    )
    api_key_field = next(
        field for field in provider.get_info().config_fields if field.id == "api_key"
    )
    assert api_key_field.required is False


def test_oauth_client_uses_bearer_auth_and_identity_headers():
    provider = AnthropicProvider(
        "sk-ant-oat-test",
        initial_auth=AnthropicAuth("sk-ant-oat-test", oauth=True),
    )
    client = provider.client
    assert client.api_key is None
    assert client.auth_token == "sk-ant-oat-test"
    assert client.default_headers["x-app"] == "cli"
    assert "oauth-2025-04-20" in client.default_headers["anthropic-beta"]


def test_native_tools_are_sent_structurally():
    provider = AnthropicProvider(
        "sk-ant-oat-test",
        initial_auth=AnthropicAuth("sk-ant-oat-test", oauth=True),
    )
    tools = provider._convert_tools_from_request(
        [
            ToolSpec(
                name="read",
                description="Read a file",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ]
    )
    params = {
        "messages": [{"role": "user", "content": "Read the README"}],
        "tools": tools,
    }
    provider._apply_oauth_request_contract(params)

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
    result = provider._convert_to_chat_response(response)

    assert params["tools"][0]["name"] == "Read"
    assert params["tools"][0]["input_schema"]["required"] == ["path"]
    assert params["messages"] == [
        {"role": "user", "content": "Read the README"}
    ]
    assert "<tools>" not in json.dumps(params["messages"])
    assert "[tool]:" not in json.dumps(params["messages"])
    assert params["system"][0]["text"].startswith("You are Claude Code")
    assert result.tool_calls[0].name == "read"
    assert result.tool_calls[0].arguments == {"path": "README.md"}
