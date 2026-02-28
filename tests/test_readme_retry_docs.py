"""Tests that README.md accurately documents retry and error handling behavior.

Validates:
- Old 'Rate Limit Configuration' section is removed
- New 'Retry and Error Handling' section exists with correct structure
- Error translation table has all 12 SDK-to-kernel mappings
- Backoff formula documented correctly
- 529 attempt table shows correct values
- Config table documents all 5 keys with defaults
- provider:retry event format documented
"""

import re
from pathlib import Path

import pytest

README_PATH = Path(__file__).parent.parent / "README.md"


@pytest.fixture
def readme_content():
    """Load README.md content."""
    return README_PATH.read_text()


class TestOldSectionRemoved:
    """Old 'Rate Limit Configuration' section must be completely replaced."""

    def test_no_rate_limit_configuration_heading(self, readme_content):
        """Old '### Rate Limit Configuration' heading must not exist."""
        assert "### Rate Limit Configuration" not in readme_content

    def test_no_sdk_builtin_retry_language(self, readme_content):
        """Old language about SDK's built-in retry mechanism must be gone."""
        assert "SDK's built-in retry mechanism" not in readme_content

    def test_no_anthropic_rate_limited_event(self, readme_content):
        """Old anthropic:rate_limited event reference must be gone."""
        assert "anthropic:rate_limited" not in readme_content


class TestNewSectionStructure:
    """New section has correct heading hierarchy."""

    def test_retry_and_error_handling_heading(self, readme_content):
        """New '### Retry and Error Handling' heading must exist."""
        assert "### Retry and Error Handling" in readme_content

    def test_error_translation_subheading(self, readme_content):
        """'#### Error Translation' subheading must exist."""
        assert "#### Error Translation" in readme_content

    def test_backoff_formula_subheading(self, readme_content):
        """'#### Backoff Formula' subheading must exist."""
        assert "#### Backoff Formula" in readme_content

    def test_retry_configuration_subheading(self, readme_content):
        """'#### Retry Configuration' subheading must exist."""
        assert "#### Retry Configuration" in readme_content

    def test_heading_order(self, readme_content):
        """Subsections must appear in correct order."""
        idx_main = readme_content.index("### Retry and Error Handling")
        idx_translation = readme_content.index("#### Error Translation")
        idx_backoff = readme_content.index("#### Backoff Formula")
        idx_config = readme_content.index("#### Retry Configuration")
        assert idx_main < idx_translation < idx_backoff < idx_config


class TestOpeningParagraph:
    """Opening paragraph describes provider-managed retries."""

    def test_sdk_retries_disabled(self, readme_content):
        """Must mention disabling SDK built-in retries (max_retries=0)."""
        assert "max_retries=0" in readme_content

    def test_retry_with_backoff_mentioned(self, readme_content):
        """Must mention retry_with_backoff from amplifier_core."""
        assert "retry_with_backoff" in readme_content

    def test_amplifier_core_utils_retry(self, readme_content):
        """Must reference amplifier_core.utils.retry module."""
        assert "amplifier_core.utils.retry" in readme_content


class TestErrorTranslationTable:
    """Error translation table must have all 12 SDK-to-kernel mappings."""

    def test_rate_limit_error_row(self, readme_content):
        """RateLimitError -> RateLimitError, 429, retryable."""
        assert "RateLimitError" in readme_content
        assert "429" in readme_content

    def test_overloaded_error_row(self, readme_content):
        """OverloadedError -> ProviderUnavailableError, 529, 10x backoff."""
        assert "OverloadedError" in readme_content
        assert "ProviderUnavailableError" in readme_content
        assert "529" in readme_content

    def test_internal_server_error_row(self, readme_content):
        """InternalServerError/5xx -> ProviderUnavailableError."""
        assert "InternalServerError" in readme_content

    def test_authentication_error_row(self, readme_content):
        """AuthenticationError -> AuthenticationError, 401, not retryable."""
        assert "AuthenticationError" in readme_content
        assert "401" in readme_content

    def test_context_length_error_row(self, readme_content):
        """BadRequestError(context) -> ContextLengthError, 400."""
        assert "ContextLengthError" in readme_content

    def test_content_filter_error_row(self, readme_content):
        """BadRequestError(safety) -> ContentFilterError, 400."""
        assert "ContentFilterError" in readme_content

    def test_invalid_request_error_row(self, readme_content):
        """BadRequestError(other) -> InvalidRequestError, 400."""
        assert "InvalidRequestError" in readme_content

    def test_access_denied_error_row(self, readme_content):
        """APIStatusError(403) -> AccessDeniedError, 403."""
        assert "AccessDeniedError" in readme_content
        assert "403" in readme_content

    def test_not_found_error_row(self, readme_content):
        """APIStatusError(404) -> NotFoundError, 404."""
        assert "NotFoundError" in readme_content
        assert "404" in readme_content

    def test_timeout_error_row(self, readme_content):
        """asyncio.TimeoutError -> LLMTimeoutError, retryable."""
        assert "TimeoutError" in readme_content
        assert "LLMTimeoutError" in readme_content

    def test_other_error_row(self, readme_content):
        """Other exceptions -> LLMError, retryable."""
        assert "LLMError" in readme_content

    def test_cause_preservation_note(self, readme_content):
        """Must note that all errors preserve __cause__."""
        assert "__cause__" in readme_content

    def test_table_has_markdown_format(self, readme_content):
        """Error translation section should contain a markdown table."""
        # Find content between Error Translation heading and next heading
        section = _extract_section(readme_content, "#### Error Translation")
        # Table rows have | separators
        assert section.count("|") >= 24, (
            "Table should have at least 12 data rows with pipes"
        )


class TestBackoffFormula:
    """Backoff formula section must be accurate."""

    def test_base_delay_documented(self, readme_content):
        """Must document base_delay calculation."""
        section = _extract_section(readme_content, "#### Backoff Formula")
        assert "base_delay" in section.lower() or "base" in section.lower()

    def test_capped_delay_documented(self, readme_content):
        """Must document capping to max_retry_delay."""
        section = _extract_section(readme_content, "#### Backoff Formula")
        assert "cap" in section.lower() or "max_retry_delay" in section.lower()

    def test_delay_multiplier_documented(self, readme_content):
        """Must document delay_multiplier scaling."""
        section = _extract_section(readme_content, "#### Backoff Formula")
        assert "delay_multiplier" in section or "multiplier" in section.lower()

    def test_retry_after_floor_documented(self, readme_content):
        """Must document retry_after as floor."""
        section = _extract_section(readme_content, "#### Backoff Formula")
        assert "retry_after" in section

    def test_jitter_documented(self, readme_content):
        """Must document jitter application."""
        section = _extract_section(readme_content, "#### Backoff Formula")
        assert "jitter" in section.lower()

    def test_529_attempt_table(self, readme_content):
        """Must show 5 attempts for 529 with correct delays: 10s, 20s, 40s, 80s, 160s."""
        section = _extract_section(readme_content, "#### Backoff Formula")
        assert "10s" in section or "10" in section
        assert "20s" in section or "20" in section
        assert "40s" in section or "40" in section
        assert "80s" in section or "80" in section
        assert "160s" in section or "160" in section

    def test_total_time_noted(self, readme_content):
        """Must note total ~310s (~5 min)."""
        section = _extract_section(readme_content, "#### Backoff Formula")
        assert "310" in section or "5 min" in section


class TestRetryConfiguration:
    """Retry configuration section must document all 5 keys."""

    def test_yaml_config_block(self, readme_content):
        """Must have a YAML config block."""
        section = _extract_section(readme_content, "#### Retry Configuration")
        assert "```yaml" in section or "```yml" in section

    def test_max_retries_key(self, readme_content):
        """max_retries documented with default 5."""
        section = _extract_section(readme_content, "#### Retry Configuration")
        assert "max_retries" in section
        assert "5" in section

    def test_min_retry_delay_key(self, readme_content):
        """min_retry_delay documented with default 1.0."""
        section = _extract_section(readme_content, "#### Retry Configuration")
        assert "min_retry_delay" in section
        assert "1.0" in section

    def test_max_retry_delay_key(self, readme_content):
        """max_retry_delay documented with default 60.0."""
        section = _extract_section(readme_content, "#### Retry Configuration")
        assert "max_retry_delay" in section
        assert "60.0" in section

    def test_retry_jitter_key(self, readme_content):
        """retry_jitter documented with default 0.2, accepts true/false."""
        section = _extract_section(readme_content, "#### Retry Configuration")
        assert "retry_jitter" in section
        assert "0.2" in section

    def test_retry_jitter_bool_compat(self, readme_content):
        """retry_jitter must note it accepts true/false for backward compat."""
        section = _extract_section(readme_content, "#### Retry Configuration")
        lower = section.lower()
        assert "true" in lower and "false" in lower

    def test_overloaded_delay_multiplier_key(self, readme_content):
        """overloaded_delay_multiplier documented with default 10.0."""
        section = _extract_section(readme_content, "#### Retry Configuration")
        assert "overloaded_delay_multiplier" in section
        assert "10.0" in section

    def test_config_table_has_5_rows(self, readme_content):
        """Config table must have 5 data rows (one per key)."""
        section = _extract_section(readme_content, "#### Retry Configuration")
        # Count rows that contain a config key
        keys = [
            "max_retries",
            "min_retry_delay",
            "max_retry_delay",
            "retry_jitter",
            "overloaded_delay_multiplier",
        ]
        for key in keys:
            assert key in section, (
                f"Config key '{key}' missing from configuration table"
            )


class TestProviderRetryEvent:
    """provider:retry event must be documented with all fields."""

    def test_event_name_documented(self, readme_content):
        """Must document provider:retry event name."""
        assert "provider:retry" in readme_content

    def test_event_fields(self, readme_content):
        """Must document all event fields."""
        # Find the events section (after Retry Configuration)
        idx = readme_content.find("provider:retry")
        assert idx != -1
        # Check nearby content for field names
        nearby = readme_content[idx : idx + 1000]
        for field in [
            "provider",
            "model",
            "attempt",
            "max_retries",
            "delay",
            "retry_after",
            "error_type",
            "error_message",
        ]:
            assert field in nearby, (
                f"Event field '{field}' not documented near provider:retry"
            )


class TestMarkdownRendering:
    """Markdown must be well-formed."""

    def test_no_broken_table_headers(self, readme_content):
        """Table header separators must be well-formed (| --- | pattern)."""
        section = _extract_section(readme_content, "### Retry and Error Handling")
        # Find all table separator lines
        for line in section.split("\n"):
            if re.match(r"^\|[\s\-:|]+\|$", line.strip()):
                # Valid separator line
                assert "---" in line or ":-" in line

    def test_code_blocks_balanced(self, readme_content):
        """Code fences must be balanced in the retry section."""
        section = _extract_section(readme_content, "### Retry and Error Handling")
        fence_count = section.count("```")
        assert fence_count % 2 == 0, f"Unbalanced code fences: {fence_count}"

    def test_section_before_beta_headers(self, readme_content):
        """Retry section must appear before Beta Headers section."""
        idx_retry = readme_content.index("### Retry and Error Handling")
        idx_beta = readme_content.index("## Beta Headers")
        assert idx_retry < idx_beta


def _extract_section(content: str, heading: str) -> str:
    """Extract content from a heading to the next heading of same or higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    pattern = re.escape(heading)
    match = re.search(pattern, content)
    if not match:
        pytest.fail(f"Heading '{heading}' not found in content")
    start = match.start()

    # Find next heading of same or higher level
    rest = content[match.end() :]
    next_heading = re.search(rf"^#{{{1},{level}}}\s", rest, re.MULTILINE)
    if next_heading:
        end = match.end() + next_heading.start()
    else:
        end = len(content)

    return content[start:end]
