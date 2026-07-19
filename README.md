# Amplifier Claude Code Provider

**Use Claude Code with Amplifier**

## Quick Start

### 1. Install Prerequisites

```bash
# Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Amplifier with Claude Provider

The easiest way is to use the provided install script, which sets up Amplifier,
registers the provider, and configures the routing matrix:

```bash
git clone https://github.com/gszep/amplifier-module-provider-claude.git
cd amplifier-module-provider-claude
bash install.sh
```

<details>
<summary>Manual installation</summary>

```bash
uv tool install git+https://github.com/microsoft/amplifier
amplifier module add provider-claude --source git+https://github.com/gszep/amplifier-module-provider-claude@main
```

If using the routing matrix, you'll also need to copy the matrix file manually.
See [Routing Matrix Integration](#routing-matrix-integration) below.

</details>

### 3. Authentication and Configuration

Authenticate directly with Anthropic using the same OAuth/PKCE flow used by Pi:

```bash
amplifier-claude-login
amplifier init  # select [3] Claude Code
```

OAuth credentials are stored at `~/.amplifier/claude-auth.json` with mode `0600`
and refreshed automatically. `ANTHROPIC_OAUTH_TOKEN` can provide an OAuth token
without the credential file. If `ANTHROPIC_API_KEY` is set, API-key billing is
used when no stored OAuth credential exists.

## Models

| Model | ID | Best For |
|-------|------|----------|
| Sonnet | `claude-sonnet-4-6` | Default — balanced speed and capability |
| Opus | `claude-opus-4-6` | Complex reasoning, extended thinking |
| Haiku | `claude-haiku-4-5` | Fast responses |

The CLI also accepts short aliases (`sonnet`, `opus`, `haiku`) for `default_model` config.

## How It Works

This provider talks directly to Anthropic's Messages API:

- Claude Pro/Max authentication uses OAuth with PKCE and automatic token refresh.
- Tool definitions are sent as native Anthropic `tools` objects.
- Tool calls and results remain native `tool_use` and `tool_result` blocks.
- Amplifier's orchestrator retains control of tool execution.
- OAuth requests carry the Claude Code identity headers required by Anthropic.

No tool definitions or calls are rendered into model-visible XML/text.

### Claude Code header compatibility

The OAuth request contract is centralized in
`amplifier_module_provider_anthropic/auth.py`. The provider discovers the installed
Claude Code version for its user-agent and has a fallback when the CLI is not
installed. `tests/test_claude_header_parity.py` checks the observable OAuth
contract against an installed Claude Code executable, while the long-running
provider integration tests verify that Anthropic accepts the request.

Claude Code does not expose final request headers and its OAuth traffic cannot be
redirected to a plaintext recorder. A true byte-for-byte capture therefore
requires a trusted TLS interception proxy. The parity test deliberately avoids
installing a local CA or exposing OAuth bearer tokens.

## Routing Matrix Integration

This provider mounts as `"claude"` (distinct from the official `"anthropic"` provider),
so both can coexist. The `install.sh` script automatically installs a
[`routing/claude.yaml`](routing/claude.yaml) matrix covering all 13 roles into the
routing-matrix bundle cache.

To activate it, set your `settings.yaml`:

```yaml
# ~/.amplifier/settings.yaml
routing:
  matrix: claude
```

The routing hook only discovers matrices from its own bundle's cache directory,
which is why `install.sh` copies the file there. If you reinstall Amplifier or
reset the cache, re-run `install.sh` to restore it.

## Contributing

This project is not currently accepting external contributions. Feel free to fork and experiment.

Most contributions require a [Contributor License Agreement](https://cla.opensource.microsoft.com).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
