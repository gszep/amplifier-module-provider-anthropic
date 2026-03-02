cd ~/.amplifier

rm -rf bundle-registry.yaml cache registry.json settings.yaml keys.env routing
cd ~

rm -rf ~/.amplifier/projects/-home-gszep/sessions/*
rm ~/.claude/projects/-home-gszep/*.jsonl

# Make sure you're not in an active virtual environment
deactivate 2>/dev/null || true

# Clear UV's cache
uv cache clean

# Uninstall Amplifier
uv tool uninstall amplifier

source ~/.bashrc

# install amplifier with claude provider
uv tool install git+https://github.com/microsoft/amplifier --with pytest-asyncio
amplifier module add provider-claude --source git+https://github.com/gszep/amplifier-module-provider-claude@main

# Patch: copy claude routing matrix into routing-matrix bundle cache.
# The routing hook only discovers matrices from its own bundle's routing/ dir,
# so we copy the provider's reference matrix there with the name settings.yaml expects.
ROUTING_CACHE=$(find ~/.amplifier/cache -maxdepth 1 -name 'amplifier-bundle-routing-matrix-*' -type d 2>/dev/null | head -1)
PROVIDER_CACHE=$(find ~/.amplifier/cache -maxdepth 1 -name 'amplifier-module-provider-claude-*' -type d 2>/dev/null | head -1)
if [ -n "$ROUTING_CACHE" ] && [ -n "$PROVIDER_CACHE" ] && [ -f "$PROVIDER_CACHE/routing/claude.yaml" ]; then
    cp "$PROVIDER_CACHE/routing/claude.yaml" "$ROUTING_CACHE/routing/claude.yaml"
    echo "Installed claude routing matrix to $ROUTING_CACHE/routing/claude.yaml"
else
    echo "Warning: Could not install claude routing matrix (cache dirs not found)"
fi

