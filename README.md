# Amplifier Claude Code Provider

**Use Claude Code with Amplifier**

## Quick Start

### 1. Install Prerequisites

```bash
# Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Claude Code CLI
curl -fsSL https://claude.ai/install.sh | bash
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

### 3. Configuration

```bash
amplifier init  # select [3] Claude Code
```
> **Note**: If `ANTHROPIC_API_KEY` is set in `~/.amplifier/keys.env` API billing will be used.

## Models

| Model | ID | Best For |
|-------|------|----------|
| Sonnet | `claude-sonnet-4-6` | Default — balanced speed and capability |
| Opus | `claude-opus-4-6` | Complex reasoning, extended thinking |
| Haiku | `claude-haiku-4-5` | Fast responses |

The CLI also accepts short aliases (`sonnet`, `opus`, `haiku`) for `default_model` config.

## How It Works

This provider wraps the Claude Code CLI:
- Tool definitions are injected via system prompt
- Claude's built-in tools are disabled (`--tools ""`)
- Amplifier's orchestrator handles all tool execution
- Responses are parsed for `[tool]:` blocks

This gives Amplifier full control over the tool ecosystem while using Claude Code.

## Routing Matrix Integration

This provider mounts as `"claude"` (distinct from the official `"anthropic"` provider),
so both can coexist. To use it with the [Routing Matrix](https://github.com/microsoft/amplifier-bundle-routing-matrix)
system, add role overrides to your `settings.yaml`:

```yaml
# ~/.amplifier/settings.yaml (or .amplifier/settings.yaml)
routing:
  matrix: anthropic   # base matrix to extend
  overrides:
    general:
      candidates:
        - provider: claude
          model: claude-sonnet-4-6
        - base   # fall back to the base matrix's candidates
    fast:
      candidates:
        - provider: claude
          model: claude-haiku-4-5
        - base
    coding:
      candidates:
        - provider: claude
          model: claude-sonnet-4-6
        - base
    reasoning:
      candidates:
        - provider: claude
          model: claude-opus-4-6
          config:
            reasoning_effort: high
        - base
```

The `base` keyword appends the original matrix's candidates for each role, giving
automatic fallback to the direct API provider if the CLI is unavailable.

A complete reference matrix covering all 13 roles is available at
[`routing/claude.yaml`](routing/claude.yaml).

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — Prompt structure, text-based tool calling, session caching
- **[Feature Coverage](docs/FEATURE_COVERAGE.md)** — Comparison with the Anthropic API provider, known limitations

## Contributing

This project is not currently accepting external contributions. Feel free to fork and experiment.

Most contributions require a [Contributor License Agreement](https://cla.opensource.microsoft.com).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
