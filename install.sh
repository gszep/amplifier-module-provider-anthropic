#!/usr/bin/env bash
set -euo pipefail

SOURCE="git+https://github.com/gszep/amplifier-module-provider-claude@main"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if ! command -v amplifier >/dev/null 2>&1; then
  uv tool install git+https://github.com/microsoft/amplifier
fi

amplifier module add provider-anthropic --source "$SOURCE"

echo "Installed the OAuth-enabled provider as the drop-in 'anthropic' provider."
echo "Run amplifier-anthropic-login to authenticate a Claude Pro/Max account."
