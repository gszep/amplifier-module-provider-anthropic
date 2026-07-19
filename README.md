# Amplifier Anthropic Provider — Claude Pro/Max OAuth Fork

This is a drop-in fork of
[`microsoft/amplifier-module-provider-anthropic`](https://github.com/microsoft/amplifier-module-provider-anthropic).
It keeps the official provider implementation and adds Anthropic Claude Pro/Max
OAuth authentication using the same direct Messages API approach as Pi.

The provider remains registered as **`anthropic`**. Existing bundles, routing
configuration, model names, API-key authentication, streaming, tools, thinking,
caching, retries, rate-limit handling, and cost tracking continue to use the
official provider implementation.

## Installation

Install this branch in place of the official provider:

```bash
amplifier module add provider-anthropic \
  --source git+https://github.com/gszep/amplifier-module-provider-claude@main
```

The repository can be renamed to `amplifier-module-provider-anthropic` without
changing the Python package or provider identity.

## Claude Pro/Max Login

```bash
amplifier-anthropic-login
```

`amplifier-claude-login` is retained as an alias.

OAuth credentials are stored in `~/.amplifier/claude-auth.json` with mode
`0600` and refreshed automatically. Authentication precedence is:

1. `ANTHROPIC_OAUTH_TOKEN`
2. Stored OAuth credentials
3. Configured `api_key`
4. `ANTHROPIC_API_KEY`

API-key users therefore retain the official provider behavior when no OAuth
credential is present.

## Native tool calling

Requests go directly through the official provider's Anthropic Messages API
transport:

- tool definitions use native `tools[].input_schema` objects;
- calls use native `tool_use` blocks;
- results use native `tool_result` blocks;
- Amplifier remains responsible for executing tools.

No tool definitions or calls are serialized into model-visible XML or
`[tool]: {...}` text.

## OAuth request contract

For OAuth requests, the fork adds the Claude Code request contract used by Pi:

- bearer-token authentication (`auth_token`, not `x-api-key`);
- Claude Code and OAuth beta headers;
- Claude Code `User-Agent` and `x-app: cli` identity headers;
- the Claude Code identity system block;
- Claude Code casing for matching built-in tool names.

The contract is centralized in
[`amplifier_module_provider_anthropic/auth.py`](amplifier_module_provider_anthropic/auth.py).
The installed Claude Code version is used in the user-agent when available.

[`tests/test_claude_header_parity.py`](tests/test_claude_header_parity.py)
checks the observable contract against an installed Claude Code executable.
Claude Code does not expose its final headers, so a byte-for-byte comparison
would require a trusted TLS interception proxy and risk exposing OAuth bearer
tokens; the test deliberately avoids that. Long-running provider integration
tests verify that Anthropic accepts the resulting request.

## Maintaining the fork

`upstream` should point to Microsoft's repository:

```bash
git remote add upstream git@github.com:microsoft/amplifier-module-provider-anthropic.git
git fetch upstream
git rebase upstream/main
```

Keep the OAuth changes as a small patch series over `upstream/main`. This makes
upstream provider updates and security fixes straightforward to consume.

## Development

```bash
uv sync --group dev
uv run pytest
uv run pytest -m long
```

## License

MIT. See [LICENSE](LICENSE).
