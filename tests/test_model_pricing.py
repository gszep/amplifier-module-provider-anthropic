"""Tests for _build_pricing(): wiring _RATES into ModelInfo.pricing.

list_models() surfaces pricing previously only used internally for cost
accounting (see _cost.py / compute_cost). _build_pricing() is the pure
function that does the _RATES -> Pricing translation; tested directly here
since no existing test mocks the async client.models.list() call.
"""

from amplifier_core import Pricing
from amplifier_module_provider_anthropic import _build_pricing
from amplifier_module_provider_anthropic._cost import _RATES


class TestBuildPricing:
    """_build_pricing() translates _RATES entries into Pricing objects."""

    def test_known_model_returns_populated_pricing(self):
        pricing = _build_pricing("claude-sonnet-4-5-20250929")

        assert pricing is not None
        assert isinstance(pricing, Pricing)
        assert pricing.input_per_million > 0
        assert pricing.output_per_million > 0
        assert pricing.currency == "USD"

    def test_pricing_matches_rates_table(self):
        rates = _RATES["claude-sonnet-4-5-20250929"]
        pricing = _build_pricing("claude-sonnet-4-5-20250929")

        assert pricing is not None
        assert pricing.input_per_million == float(rates["input_per_m"])
        assert pricing.output_per_million == float(rates["output_per_m"])
        assert pricing.cache_read_per_million == float(rates["cache_read_per_m"])
        assert pricing.cache_write_per_million == float(rates["cache_write_per_m"])

    def test_unknown_model_returns_none(self):
        assert _build_pricing("claude-mystery-9-9") is None

    def test_all_rate_table_entries_build_valid_pricing(self):
        """Every model in _RATES should produce a valid Pricing object."""
        for model_id in _RATES:
            pricing = _build_pricing(model_id)
            assert pricing is not None, f"Expected pricing for {model_id}"
            assert pricing.input_per_million > 0
            assert pricing.output_per_million > 0
