"""Drift checks against an installed Claude Code executable.

Claude Code does not expose its final request headers, and OAuth traffic cannot
safely be redirected to a plaintext recorder. This test checks the observable
identity contract in the installed executable. The live integration tests then
verify that Anthropic accepts the resulting provider request.
"""

from pathlib import Path
import re
import shutil
import subprocess

import pytest

from amplifier_module_provider_anthropic.auth import (
    OAUTH_BETAS,
    installed_claude_code_version,
    oauth_request_headers,
)


@pytest.mark.long
def test_installed_claude_code_oauth_header_contract():
    executable = shutil.which("claude")
    if not executable:
        pytest.skip("Claude Code is not installed")

    resolved = Path(executable).resolve()
    binary = resolved.read_bytes()
    headers = oauth_request_headers()
    version = installed_claude_code_version()

    reported = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout
    assert re.search(rf"\b{re.escape(version)}\b", reported)
    expected_user_agent = f"claude-cli/{version} (external, cli)"
    assert headers["user-agent"] == expected_user_agent
    assert b"claude-cli/" in binary
    assert b"(external, " in binary
    assert b"x-app" in binary

    # The OAuth beta is the authentication protocol contract. Tool-streaming
    # betas are provider capabilities and need not all be embedded by the CLI.
    oauth_beta = next(beta for beta in OAUTH_BETAS if beta.startswith("oauth-"))
    assert oauth_beta.encode() in binary, (
        f"Installed Claude Code no longer contains {oauth_beta}; inspect its "
        "OAuth request contract before updating OAUTH_BETAS"
    )
