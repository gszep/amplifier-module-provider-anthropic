cd ~/.amplifier

rm -rf bundle-registry.yaml cache registry.json settings.yaml keys.env routing
cd ~

rm -rf ~/.amplifier/projects/-home-gszep/sessions/*
rm ~/.claude/projects/-home-gszep/*.jsonl

deactivate 2>/dev/null || true
uv cache clean
uv tool uninstall amplifier

source ~/.bashrc

uv tool install git+https://github.com/microsoft/amplifier --with pytest-asyncio
amplifier module add provider-claude --source git+https://github.com/gszep/amplifier-module-provider-claude@main
ROUTING_CACHE=$(find ~/.amplifier/cache -maxdepth 1 -name 'amplifier-bundle-routing-matrix-*' -type d 2>/dev/null | head -1)
PROVIDER_CACHE=$(find ~/.amplifier/cache -maxdepth 1 -name 'amplifier-module-provider-claude-*' -type d 2>/dev/null | head -1)
if [ -n "$ROUTING_CACHE" ] && [ -n "$PROVIDER_CACHE" ] && [ -f "$PROVIDER_CACHE/routing/claude.yaml" ]; then
    cp "$PROVIDER_CACHE/routing/claude.yaml" "$ROUTING_CACHE/routing/claude.yaml"
    echo "Installed claude routing matrix to $ROUTING_CACHE/routing/claude.yaml"
else
    echo "Warning: Could not install claude routing matrix (cache dirs not found)"
fi

