"""Tests for cross-process shared rate-limit state file (Option A+).

Seven tests covering:

  1. Write creates atomic JSON file with correct fields
  2. Read loads and merges state (takes lower remaining values)
  3. Stale data (>120s old) is ignored
  4. Missing/corrupt file doesn't crash
  5. Disabled when path is empty string
  6. Write is debounced (same data doesn't re-write)
  7. Read is cached (not re-read within 1 second)
"""

import json
import os
import time

import pytest

from amplifier_module_provider_anthropic import AnthropicProvider
from amplifier_module_provider_anthropic import _RateLimitState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_HEADERS: dict = {
    "requests_remaining": 40,
    "requests_limit": 100,
    "requests_reset": "2026-03-03T20:15:00Z",
    "input_tokens_remaining": 450_000,
    "input_tokens_limit": 1_000_000,
    "input_tokens_reset": "2026-03-03T20:15:00Z",
    "output_tokens_remaining": 90_000,
    "output_tokens_limit": 200_000,
    "output_tokens_reset": "2026-03-03T20:15:00Z",
}


def _make_provider(tmp_path, **extra_config) -> AnthropicProvider:
    """Create a provider pointing its shared state file at a temp directory."""
    state_path = str(tmp_path / "rate-limit-state.json")
    config = {
        "use_streaming": False,
        "max_retries": 0,
        "rate_limit_state_path": state_path,
        **extra_config,
    }
    return AnthropicProvider(api_key="test-key", config=config)


# ---------------------------------------------------------------------------
# 1. Write creates atomic JSON file with correct fields
# ---------------------------------------------------------------------------


class TestWriteSharedState:
    def test_creates_json_file_with_all_rate_limit_fields(self, tmp_path):
        """Write should create the JSON file with rate-limit fields + metadata."""
        provider = _make_provider(tmp_path)
        provider._write_shared_rate_limit_state(_SAMPLE_HEADERS)

        path = tmp_path / "rate-limit-state.json"
        assert path.exists(), "Expected the state file to be created"

        data = json.loads(path.read_text())
        assert "updated_at" in data
        assert data["updated_by_pid"] == os.getpid()
        assert data["requests_remaining"] == 40
        assert data["requests_limit"] == 100
        assert data["input_tokens_remaining"] == 450_000
        assert data["output_tokens_remaining"] == 90_000

    def test_tmp_file_does_not_exist_after_write(self, tmp_path):
        """The .tmp scratch file should be renamed away — not left behind."""
        provider = _make_provider(tmp_path)
        provider._write_shared_rate_limit_state(_SAMPLE_HEADERS)

        tmp_file = tmp_path / "rate-limit-state.json.tmp"
        assert not tmp_file.exists(), "Temp file should be renamed, not left behind"

    def test_write_omits_none_fields(self, tmp_path):
        """Fields that are None in the rate_limit_info dict should not appear."""
        provider = _make_provider(tmp_path)
        partial = {"requests_remaining": 10, "requests_limit": 50}
        provider._write_shared_rate_limit_state(partial)

        data = json.loads((tmp_path / "rate-limit-state.json").read_text())
        assert "requests_remaining" in data
        assert "input_tokens_remaining" not in data  # not provided → omitted


# ---------------------------------------------------------------------------
# 2. Read loads and merges state (takes lower remaining values)
# ---------------------------------------------------------------------------


class TestReadSharedState:
    def test_merge_takes_lower_remaining(self, tmp_path):
        """Read should merge in the LOWER remaining value, not blindly overwrite."""
        provider = _make_provider(tmp_path)
        # Seed the local state with high remaining values
        provider._rate_limit_state.update_from_headers(
            {
                "requests_remaining": 80,
                "requests_limit": 100,
                "input_tokens_remaining": 900_000,
                "input_tokens_limit": 1_000_000,
                "output_tokens_remaining": 180_000,
                "output_tokens_limit": 200_000,
            }
        )

        # Shared file shows that another process has consumed more capacity
        shared = {
            "updated_at": time.time(),
            "updated_by_pid": 99999,
            "requests_remaining": 20,
            "requests_limit": 100,
            "input_tokens_remaining": 300_000,
            "input_tokens_limit": 1_000_000,
            "output_tokens_remaining": 60_000,
            "output_tokens_limit": 200_000,
        }
        (tmp_path / "rate-limit-state.json").write_text(json.dumps(shared))

        provider._read_shared_rate_limit_state()

        # Local values should have been lowered to the shared minimums
        assert provider._rate_limit_state.requests_remaining == 20
        assert provider._rate_limit_state.input_tokens_remaining == 300_000
        assert provider._rate_limit_state.output_tokens_remaining == 60_000

    def test_merge_keeps_local_when_local_is_lower(self, tmp_path):
        """When local remaining is already lower, it must not be raised by shared."""
        provider = _make_provider(tmp_path)
        # Local is critically low
        provider._rate_limit_state.update_from_headers(
            {"requests_remaining": 2, "requests_limit": 100}
        )
        # Shared file looks healthier
        shared = {
            "updated_at": time.time(),
            "updated_by_pid": 99999,
            "requests_remaining": 50,
            "requests_limit": 100,
        }
        (tmp_path / "rate-limit-state.json").write_text(json.dumps(shared))

        provider._read_shared_rate_limit_state()

        # Local must stay at 2 — shared must not override upward
        assert provider._rate_limit_state.requests_remaining == 2

    def test_read_adopts_shared_when_local_has_no_data(self, tmp_path):
        """When local state is blank, the shared file values are adopted."""
        provider = _make_provider(tmp_path)
        assert provider._rate_limit_state.requests_remaining is None

        shared = {
            "updated_at": time.time(),
            "updated_by_pid": 99999,
            "requests_remaining": 55,
            "requests_limit": 100,
        }
        (tmp_path / "rate-limit-state.json").write_text(json.dumps(shared))

        provider._read_shared_rate_limit_state()

        assert provider._rate_limit_state.requests_remaining == 55


# ---------------------------------------------------------------------------
# 3. Stale data (>120s old) is ignored
# ---------------------------------------------------------------------------


class TestStaleness:
    def test_stale_data_is_ignored(self, tmp_path):
        """Shared state older than 120 seconds must not affect local state."""
        provider = _make_provider(tmp_path)
        stale = {
            "updated_at": time.time() - 121,  # 121 s ago → stale
            "updated_by_pid": 99999,
            "requests_remaining": 5,
            "requests_limit": 100,
        }
        (tmp_path / "rate-limit-state.json").write_text(json.dumps(stale))

        provider._read_shared_rate_limit_state()

        # Stale data should NOT have been merged
        assert provider._rate_limit_state.requests_remaining is None

    def test_fresh_data_within_120s_is_used(self, tmp_path):
        """Data that is 119 seconds old (just inside the window) should be used."""
        provider = _make_provider(tmp_path)
        fresh_enough = {
            "updated_at": time.time() - 119,
            "updated_by_pid": 99999,
            "requests_remaining": 10,
            "requests_limit": 100,
        }
        (tmp_path / "rate-limit-state.json").write_text(json.dumps(fresh_enough))

        provider._read_shared_rate_limit_state()

        assert provider._rate_limit_state.requests_remaining == 10


# ---------------------------------------------------------------------------
# 4. Missing/corrupt file doesn't crash
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_missing_file_does_not_crash(self, tmp_path):
        """Read must succeed silently when the file doesn't exist."""
        provider = _make_provider(tmp_path)
        # No file created — should be a no-op
        provider._read_shared_rate_limit_state()  # must not raise

    def test_corrupt_json_does_not_crash(self, tmp_path):
        """Read must succeed silently when the file contains garbage."""
        provider = _make_provider(tmp_path)
        (tmp_path / "rate-limit-state.json").write_text("{bad json!!!")

        provider._read_shared_rate_limit_state()  # must not raise
        assert provider._rate_limit_state.requests_remaining is None

    def test_write_to_unwritable_path_does_not_crash(self, tmp_path):
        """Write must succeed silently when it cannot create the file."""
        provider = _make_provider(tmp_path)
        # Point at a path where the parent is a file (can't mkdir)
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a dir")
        provider._shared_state_path = str(blocker / "state.json")

        provider._write_shared_rate_limit_state(_SAMPLE_HEADERS)  # must not raise


# ---------------------------------------------------------------------------
# 5. Disabled when path is empty string
# ---------------------------------------------------------------------------


class TestDisabled:
    def test_write_is_noop_when_path_empty(self, tmp_path):
        """No file should be created when rate_limit_state_path is ''."""
        provider = _make_provider(tmp_path, rate_limit_state_path="")
        provider._write_shared_rate_limit_state(_SAMPLE_HEADERS)

        # Nothing should be written anywhere
        assert list(tmp_path.iterdir()) == []

    def test_read_is_noop_when_path_empty(self, tmp_path):
        """Read should be a no-op and not change local state when path is ''."""
        provider = _make_provider(tmp_path, rate_limit_state_path="")
        provider._read_shared_rate_limit_state()  # must not raise
        assert provider._rate_limit_state.requests_remaining is None


# ---------------------------------------------------------------------------
# 6. Write is debounced (same data doesn't re-write)
# ---------------------------------------------------------------------------


class TestWriteDebounce:
    def test_identical_data_skips_second_write(self, tmp_path):
        """Writing the same rate-limit data twice should only produce one file write."""
        provider = _make_provider(tmp_path)
        path = tmp_path / "rate-limit-state.json"

        provider._write_shared_rate_limit_state(_SAMPLE_HEADERS)
        assert path.exists()
        mtime_after_first = path.stat().st_mtime_ns

        # Force a tiny sleep so mtime would differ if a write occurred
        time.sleep(0.02)

        provider._write_shared_rate_limit_state(_SAMPLE_HEADERS)
        mtime_after_second = path.stat().st_mtime_ns

        assert mtime_after_first == mtime_after_second, (
            "File should not be rewritten when data hasn't changed"
        )

    def test_changed_data_triggers_new_write(self, tmp_path):
        """A different remaining value must bypass the debounce and write."""
        provider = _make_provider(tmp_path)
        path = tmp_path / "rate-limit-state.json"

        provider._write_shared_rate_limit_state(_SAMPLE_HEADERS)
        time.sleep(0.02)

        updated = {**_SAMPLE_HEADERS, "requests_remaining": 39}  # one less
        provider._write_shared_rate_limit_state(updated)

        data = json.loads(path.read_text())
        assert data["requests_remaining"] == 39


# ---------------------------------------------------------------------------
# 7. Read is cached (not re-read within 1 second)
# ---------------------------------------------------------------------------


class TestReadCache:
    def test_second_read_within_1s_is_skipped(self, tmp_path):
        """A second _read call within 1 second must not re-parse the file."""
        provider = _make_provider(tmp_path)

        first_state = {
            "updated_at": time.time(),
            "updated_by_pid": 99999,
            "requests_remaining": 77,
            "requests_limit": 100,
        }
        state_file = tmp_path / "rate-limit-state.json"
        state_file.write_text(json.dumps(first_state))

        # First read — should apply the file
        provider._read_shared_rate_limit_state()
        assert provider._rate_limit_state.requests_remaining == 77

        # Now update the file — but read again immediately (cache should block)
        updated_state = {**first_state, "requests_remaining": 10}
        state_file.write_text(json.dumps(updated_state))

        provider._read_shared_rate_limit_state()  # should be a cache hit
        # Value must still be 77 (the cached read), not 10
        assert provider._rate_limit_state.requests_remaining == 77

    def test_read_after_1s_re_reads_file(self, tmp_path):
        """After 1 second, _read must pick up new file content."""
        provider = _make_provider(tmp_path)

        state_file = tmp_path / "rate-limit-state.json"
        state_file.write_text(
            json.dumps(
                {
                    "updated_at": time.time(),
                    "updated_by_pid": 99999,
                    "requests_remaining": 77,
                    "requests_limit": 100,
                }
            )
        )
        provider._read_shared_rate_limit_state()
        assert provider._rate_limit_state.requests_remaining == 77

        # Simulate that 1.1 seconds have passed by backdating the last-read timestamp
        provider._last_shared_state_read -= 1.1

        # Update the file
        state_file.write_text(
            json.dumps(
                {
                    "updated_at": time.time(),
                    "updated_by_pid": 99999,
                    "requests_remaining": 10,
                    "requests_limit": 100,
                }
            )
        )

        provider._read_shared_rate_limit_state()
        assert provider._rate_limit_state.requests_remaining == 10
