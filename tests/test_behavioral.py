"""Behavioral tests for anthropic provider.

Inherits authoritative tests from amplifier-core.
"""

import pytest
from amplifier_core.validation.behavioral import ProviderBehaviorTests


class TestAnthropicProviderBehavior(ProviderBehaviorTests):
    """Run standard provider behavioral tests for anthropic.

    All tests from ProviderBehaviorTests run automatically.
    Add module-specific tests below if needed.
    """

    @pytest.mark.skip(reason="Anthropic's model catalog requires live credentials")
    async def test_list_models_returns_list(self, provider_module):
        pass
