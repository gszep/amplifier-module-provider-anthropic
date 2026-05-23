"""Claude Code CLI provider module for Amplifier."""

__all__ = ["mount", "ClaudeProvider"]
__amplifier_module_type__ = "provider"
import asyncio

import json
import logging
from pathlib import Path
import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import claude_agent_sdk  # type: ignore
import claude_agent_sdk._internal.client as _sdk_internal_client  # type: ignore
import claude_agent_sdk._internal.message_parser as _sdk_message_parser  # type: ignore
from amplifier_core import (  # type: ignore
    ModelInfo,
    ModuleCoordinator,
    ProviderInfo,
    TextContent,
    ThinkingContent,
    ToolCallContent,
)
from amplifier_core.message_models import (  # type: ignore
    ChatRequest,
    ChatResponse,
    Message,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolCallBlock,
    Usage,
)
from amplifier_core.utils import redact_secrets

try:
    from amplifier_core.llm_errors import LLMError as KernelLLMError
    from amplifier_core.llm_errors import LLMTimeoutError as KernelLLMTimeoutError
    from amplifier_core.llm_errors import (
        ProviderUnavailableError as KernelProviderUnavailableError,
    )
    from amplifier_core.llm_errors import RateLimitError as KernelRateLimitError
    from amplifier_core.utils.retry import RetryConfig
    from amplifier_core.utils.retry import retry_with_backoff as _retry_with_backoff

    _HAS_RETRY = True
except ImportError:
    _HAS_RETRY = False

    class KernelLLMError(Exception):  # type: ignore[no-redef]
        def __init__(self, message: str, **kwargs: Any) -> None:
            super().__init__(message)
            self.retryable = kwargs.get("retryable", False)
            self.provider = kwargs.get("provider")
            self.model = kwargs.get("model")
            self.retry_after = kwargs.get("retry_after")
            self.delay_multiplier = kwargs.get("delay_multiplier", 1.0)

    class KernelLLMTimeoutError(KernelLLMError):  # type: ignore[no-redef]
        pass

    class KernelRateLimitError(KernelLLMError):  # type: ignore[no-redef]
        pass

    class KernelProviderUnavailableError(KernelLLMError):  # type: ignore[no-redef]
        pass

    class RetryConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            self.max_retries: int = kwargs.get("max_retries", 5)
            self.initial_delay: float = kwargs.get("initial_delay", 1.0)
            self.max_delay: float = kwargs.get("max_delay", 60.0)
            self.jitter: float = kwargs.get("jitter", 0.2)


from anthropic import AsyncAnthropic
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock  # type: ignore
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock  # type: ignore
from anthropic.types.parsed_message import (  # type: ignore
    ParsedContentBlock,
    ParsedMessage,
    ParsedTextBlock,
)
from anthropic.types.usage import Usage as AnthropicUsage  # type: ignore
from claude_agent_sdk import ClaudeSDKClient  # type: ignore
from claude_agent_sdk.types import ClaudeAgentOptions  # type: ignore
from pydantic import ValidationError

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    operation: Any,
    config: Any = None,
    *,
    on_retry: Any = None,
) -> Any:
    """Wrapper that uses amplifier-core retry when available, else single-attempt fallback."""
    if _HAS_RETRY:
        return await _retry_with_backoff(operation, config, on_retry=on_retry)
    return await operation()


# SDK patch: tolerate unknown message types (e.g. "rate_limit_event") from CLI v2.1.42+.
# Tracking: https://github.com/anthropics/claude-agent-sdk-python/issues/583
# Remove once claude-agent-sdk ships the fix (PR #589).
from claude_agent_sdk._errors import MessageParseError as _MessageParseError  # type: ignore

_original_parse_message = _sdk_message_parser.parse_message


def _tolerant_parse_message(data: dict) -> claude_agent_sdk.types.Message:
    try:
        return _original_parse_message(data)
    except _MessageParseError as exc:
        if "Unknown message type" in str(exc):
            msg_type = data.get("type", "?")
            logger.info(
                "[PROVIDER] Skipping unrecognized SDK message type: %s", msg_type
            )
            return claude_agent_sdk.types.StreamEvent(
                uuid=data.get("uuid", ""),
                session_id=data.get("session_id", ""),
                event=data,
            )
        raise  # re-raise genuine parse errors (malformed data, missing fields)


_sdk_message_parser.parse_message = _tolerant_parse_message
_sdk_internal_client.parse_message = _tolerant_parse_message
SESSION_TAG = "[session]:"
SESSION = SESSION_TAG + """{"id": null}"""


@dataclass
class WebSearchContent:
    """Content block for web search results from native Anthropic web search."""

    type: str = "web_search"
    query: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ModelCapabilities:
    """Per-model capability matrix."""

    family: str
    max_output_tokens: int = 64000
    base_context_window: int = 200000
    supports_thinking: bool = False
    supports_adaptive_thinking: bool = False
    default_thinking_budget: int = 0
    capability_tags: tuple[str, ...] = ("tools", "streaming", "json_mode")


class Session(RedactedThinkingBlock):
    """Content block for storing the session ID."""

    type: Literal["redacted_thinking"] = "redacted_thinking"
    visibility: Literal["internal"] = "internal"
    data: str = SESSION

    @property
    def json_string(self) -> str:
        return self.data.replace(SESSION_TAG, "")

    @json_string.setter
    def json_string(self, value: str):
        self.data = f"{SESSION_TAG}{value}"

    @property
    def json(self) -> dict[str, int | str]:
        return json.loads(self.json_string)

    @json.setter
    def json(self, value: dict[str, int | str]):
        self.json_string = json.dumps(value)

    @property
    def id(self) -> str | None:
        return self.json.get("id", None)

    @id.setter
    def id(self, value: str | None):
        self.json |= {"id": value}


# CLI alias -> (full model ID, display name)
_CANONICAL_MODELS: dict[str, tuple[str, str]] = {
    "opus": ("claude-opus-4-6", "Claude Opus 4.6"),
    "sonnet": ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    "haiku": ("claude-haiku-4-5", "Claude Haiku 4.5"),
}


class ClaudeChatResponse(ChatResponse):
    content_blocks: (
        list[
            TextContent | ThinkingContent | ToolCallContent | WebSearchContent | Session
        ]
        | None
    ) = None
    text: str | None = None
    web_search_results: list[dict[str, Any]] | None = None


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """Mount the Claude provider."""
    config = config or {}

    cli_path = shutil.which("claude")
    if not cli_path:
        logger.warning(
            "Claude Code CLI not found. Install with: "
            "curl -fsSL https://claude.ai/install.sh | bash"
        )
        return None

    # Warn if Claude CLI context compaction is not disabled globally.
    # Compaction strips the <tools> block, causing tool-call hallucinations.
    claude_json = Path.home() / ".claude.json"
    try:
        if claude_json.exists():
            claude_cfg = json.loads(claude_json.read_text())
            if claude_cfg.get("autoCompactEnabled") is not False:
                logger.warning(
                    "[PROVIDER] Claude CLI autoCompactEnabled is not disabled "
                    "in %s. Context compaction can strip tool definitions and "
                    "cause tool-call failures. Set '\"autoCompactEnabled\": false' "
                    "in %s to prevent this.",
                    claude_json,
                    claude_json,
                )
        else:
            logger.info(
                "[PROVIDER] %s not found — cannot verify autoCompactEnabled setting.",
                claude_json,
            )
    except Exception as exc:
        logger.debug("[PROVIDER] Failed to read %s: %s", claude_json, exc)

    provider = ClaudeProvider(config=config, coordinator=coordinator)
    await coordinator.mount("providers", provider, name="claude")
    logger.info("Mounted ClaudeProvider")

    async def cleanup():
        await provider.close()

    return cleanup


class ClaudeProvider:
    """Claude Code CLI integration for Amplifier."""

    name = "claude"
    api_label = "Claude Code"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        coordinator: ModuleCoordinator | None = None,
    ):
        self.config = config or {}
        self.coordinator = coordinator

        self.default_model: str = self.config.get("default_model", "sonnet")
        self.temperature: float = self.config.get("temperature", 1.0)  # thinking
        self.enable_prompt_caching: bool = self.config.get(
            "enable_prompt_caching", True
        )

        self._beta_headers: list[str] = []
        self.context_window = 200000

        default_caps = self._get_capabilities(self.default_model)
        self.max_tokens: int = self.config.get(
            "max_tokens", default_caps.max_output_tokens
        )
        self.max_output_tokens: int = default_caps.max_output_tokens
        self.max_thinking_tokens: int = self.config.get(
            "max_thinking_tokens", default_caps.default_thinking_budget
        )

        self.priority: int = self.config.get("priority", 100)
        self.raw: bool = self.config.get("raw", False)

        self.timeout: float = self.config.get("timeout", 600.0)
        self.use_streaming = True
        self.enable_web_search: bool = self.config.get("enable_web_search", False)

        self._repaired_tool_ids: set[str] = set()
        self._available_tool_names: set[str] = set()
        self._last_tools_text: str = ""
        self._session: Session = Session()
        self._client: AsyncAnthropic | None = None

        self._retry_config = RetryConfig(
            max_retries=self.config.get("max_retries", 5),
            initial_delay=self.config.get("min_retry_delay", 1.0),
            max_delay=self.config.get("max_retry_delay", 60.0),
            jitter=bool(self.config.get("retry_jitter", True)),
        )
        self._overloaded_delay_multiplier: float = self.config.get(
            "overloaded_delay_multiplier", 10.0
        )

    @property
    def client(self) -> AsyncAnthropic:
        if self._client is None:
            self._client = AsyncAnthropic(max_retries=0)
        return self._client

    def get_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="claude",
            display_name="Claude Code",
            credential_env_vars=[],
            capabilities=list(
                self._get_capabilities(self.default_model).capability_tags
            ),
            defaults={
                "model": self.default_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "timeout": self.timeout,
                "context_window": self.context_window,
                "max_output_tokens": self.max_output_tokens,
            },
            config_fields=[],
        )

    async def close(self) -> None:
        """Close the underlying Anthropic client to prevent resource leaks."""
        if self._client is not None:
            try:
                await asyncio.shield(self._client.close())
            except asyncio.CancelledError:
                pass

    async def list_models(self) -> list[ModelInfo]:
        result = []
        for alias, (canonical_id, display_name) in _CANONICAL_MODELS.items():
            caps = self._get_capabilities(canonical_id)
            result.append(
                ModelInfo(
                    id=canonical_id,
                    display_name=display_name,
                    context_window=self.context_window,
                    max_output_tokens=caps.max_output_tokens,
                    capabilities=list(caps.capability_tags),
                    defaults={
                        "temperature": self.temperature,
                        "max_tokens": caps.max_output_tokens,
                    },
                )
            )
        return result

    @staticmethod
    def _detect_family(model_id: str) -> str:
        model_lower = model_id.lower()
        for family in ("opus", "sonnet", "haiku"):
            if family in model_lower:
                return family
        return "sonnet"  # Default to sonnet for unknown models

    @staticmethod
    def _detect_version(model_id: str, family: str) -> tuple[int, int]:
        # Match e.g. "claude-opus-4-6" (major-minor) but NOT
        # "claude-opus-4-20250514" where the second segment is a snapshot date.
        # The {1,2} bound on minor prevents accidental matches against date
        # suffixes which would incorrectly classify a model as a high version
        # (e.g. (4, 20250514) >= (4, 7) is True).
        pattern = rf"{family}-(\d+)-(\d{{1,2}})(?:-|$)"
        match = re.search(pattern, model_id.lower())
        if match:
            return int(match.group(1)), int(match.group(2))
        # Fall back to major-only when no minor is detectable; treat minor
        # as unknown (0) so capability tiers default to the conservative path.
        major_only_pattern = rf"{family}-(\d+)(?:-|$)"
        match = re.search(major_only_pattern, model_id.lower())
        if match:
            return int(match.group(1)), 0
        return (0, 0)

    @classmethod
    def _get_capabilities(cls, model_id: str) -> ModelCapabilities:
        family = cls._detect_family(model_id)
        major, minor = cls._detect_version(model_id, family)
        version_known = (major, minor) != (0, 0)

        if family == "opus":
            is_46_plus = not version_known or (major, minor) >= (4, 6)
            return ModelCapabilities(
                family="opus",
                max_output_tokens=128000 if is_46_plus else 64000,
                supports_thinking=True,
                supports_adaptive_thinking=is_46_plus,
                default_thinking_budget=64000 if is_46_plus else 32000,
                capability_tags=(
                    "tools",
                    "thinking",
                    "streaming",
                    "json_mode",
                    "vision",
                ),
            )

        if family == "sonnet":
            return ModelCapabilities(
                family="sonnet",
                supports_thinking=True,
                supports_adaptive_thinking=False,
                default_thinking_budget=32000,
                capability_tags=(
                    "tools",
                    "thinking",
                    "streaming",
                    "json_mode",
                    "vision",
                ),
            )

        if family == "haiku":
            is_45_plus = not version_known or (major, minor) >= (4, 5)
            if is_45_plus:
                return ModelCapabilities(
                    family="haiku",
                    supports_thinking=True,
                    supports_adaptive_thinking=False,
                    default_thinking_budget=32000,
                    capability_tags=(
                        "tools",
                        "thinking",
                        "streaming",
                        "json_mode",
                        "fast",
                        "vision",
                    ),
                )
            return ModelCapabilities(
                family="haiku",
                capability_tags=("tools", "streaming", "json_mode", "fast", "vision"),
            )

        return ModelCapabilities(family=family)

    @staticmethod
    def _has_session_block(msg: Message) -> bool:
        if msg.role != "assistant" or not isinstance(msg.content, list):
            return False
        for block in msg.content:
            if (
                hasattr(block, "type")
                and block.type == "redacted_thinking"
                and hasattr(block, "data")
                and isinstance(block.data, str)
                and block.data.startswith(SESSION_TAG)
            ):
                return True
        return False

    def _get_recent_messages(self, messages: list[Message]) -> list[Message]:
        """Subset messages for incremental CLI delivery using session block anchor."""
        has_system_prefix = bool(messages) and messages[0].role == "system"
        if not has_system_prefix:
            return messages

        system_end = 0
        for i, m in enumerate(messages):
            if m.role == "system":
                system_end = i + 1
            else:
                break

        system_messages = messages[:system_end]
        conversation = messages[system_end:]

        if self._session.id:
            anchor_idx = None
            for i in range(len(conversation) - 1, -1, -1):
                if self._has_session_block(conversation[i]):
                    anchor_idx = i
                    break

            if anchor_idx is not None:
                conversation = conversation[anchor_idx:]

        return list(system_messages) + list(conversation)

    def _find_missing_tool_results(
        self, messages: list[Message]
    ) -> list[tuple[int, str, str, dict]]:
        tool_calls = {}  # {call_id: (msg_index, name, args)}
        tool_results = set()  # {call_id}

        for idx, msg in enumerate(messages):
            if msg.role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "tool_call":
                        tool_calls[block.id] = (idx, block.name, block.input)

            elif (
                msg.role == "tool" and hasattr(msg, "tool_call_id") and msg.tool_call_id
            ):
                tool_results.add(msg.tool_call_id)

        return [
            (msg_idx, call_id, name, args)
            for call_id, (msg_idx, name, args) in tool_calls.items()
            if call_id not in tool_results and call_id not in self._repaired_tool_ids
        ]

    def _create_synthetic_result(self, call_id: str, tool_name: str) -> Message:
        return Message(
            role="tool",
            content=(
                f"[SYSTEM ERROR: Tool result missing from conversation history]\n\n"
                f"Tool: {tool_name}\n"
                f"Call ID: {call_id}\n\n"
                f"This indicates the tool result was lost after execution.\n"
                f"Likely causes: context compaction bug, message parsing error, or state corruption.\n\n"
                f"The tool may have executed successfully, but the result was lost.\n"
                f"Please acknowledge this error and offer to retry the operation."
            ),
            tool_call_id=call_id,
            name=tool_name,
        )

    async def complete(self, request: ChatRequest, **kwargs: Any) -> ChatResponse:
        missing = self._find_missing_tool_results(request.messages)

        if missing:
            logger.warning(
                f"[PROVIDER] Anthropic: Detected {len(missing)} missing tool result(s). "
                f"Injecting synthetic errors. This indicates a bug in context management. "
                f"Tool IDs: {[call_id for _, call_id, _, _ in missing]}"
            )

            from collections import defaultdict

            by_msg_idx: dict[int, list[tuple[str, str]]] = defaultdict(list)
            for msg_idx, call_id, tool_name, _ in missing:
                by_msg_idx[msg_idx].append((call_id, tool_name))

            for msg_idx in sorted(by_msg_idx.keys(), reverse=True):
                synthetics = []
                for call_id, tool_name in by_msg_idx[msg_idx]:
                    synthetics.append(self._create_synthetic_result(call_id, tool_name))
                    self._repaired_tool_ids.add(call_id)

                insert_pos = msg_idx + 1
                for i, synthetic in enumerate(synthetics):
                    request.messages.insert(insert_pos + i, synthetic)

            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:tool_sequence_repaired",
                    {
                        "provider": self.name,
                        "repair_count": len(missing),
                        "repairs": [
                            {"tool_call_id": call_id, "tool_name": tool_name}
                            for _, call_id, tool_name, _ in missing
                        ],
                    },
                )

        response = await self._complete_chat_request(request, **kwargs)

        # Handle provider-internal list_tools calls before the orchestrator sees them.
        if response.tool_calls and any(
            tc.name == "list_tools" for tc in response.tool_calls
        ):
            response = await self._fulfill_list_tools(request, response, **kwargs)

        return response

    async def _fulfill_list_tools(
        self,
        request: ChatRequest,
        response: ChatResponse,
        **kwargs: Any,
    ) -> ChatResponse:
        """Intercept list_tools calls: inject tool definitions and re-send."""
        logger.info(
            "[PROVIDER] Model called list_tools — re-injecting tool definitions"
        )

        # Build assistant message from the current response
        assistant_content: list[Any] = []
        tool_calls_dicts: list[dict[str, Any]] = []

        if response.content:
            for block in response.content:
                # Pass through all content blocks (text, thinking, session, etc.)
                assistant_content.append(block)

        for tc in response.tool_calls or []:
            if not any(
                hasattr(b, "name")
                and b.name == tc.name
                and hasattr(b, "id")
                and b.id == tc.id
                for b in assistant_content
            ):
                assistant_content.append(
                    ToolCallBlock(id=tc.id, name=tc.name, input=tc.arguments or {})
                )
            tool_calls_dicts.append(
                {"id": tc.id, "tool": tc.name, "arguments": tc.arguments or {}}
            )

        request.messages.append(
            Message(
                role="assistant",
                content=assistant_content,
                tool_calls=tool_calls_dicts,
            )
        )

        # Append tool results for every tool call in this turn
        tools_text = self._last_tools_text or "No tool definitions available."
        for tc in response.tool_calls or []:
            if tc.name == "list_tools":
                request.messages.append(
                    Message(
                        role="tool",
                        content=tools_text,
                        tool_call_id=tc.id,
                        name="list_tools",
                    )
                )
            else:
                # Provide synthetic placeholder for concurrent tool calls
                request.messages.append(
                    Message(
                        role="tool",
                        content=(
                            f"[Deferred: {tc.name} dispatch pending — "
                            f"tool definitions were refreshed. Please re-call this tool.]"
                        ),
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )

        return await self._complete_chat_request(request, **kwargs)

    def _format_tools_listing(self, tools: list[dict[str, Any]]) -> str:
        """Format tool definitions for list_tools output."""
        parts = []
        for t in tools:
            name = t.get("name", "?")
            schema = json.dumps(t.get("input_schema", {}))
            desc = t.get("description", "")
            parts.append(
                f'{{"name": "{name}", "input_schema": {schema}}}'
                f"\n<instructions>\n{desc}\n</instructions>"
            )
        return "<tools>\n" + "\n\n".join(parts) + "\n</tools>"

    def _format_system_with_cache(
        self, system_msgs: list[Message]
    ) -> list[dict[str, Any]] | None:
        if not system_msgs:
            return None

        combined = "\n\n".join(
            m.content if isinstance(m.content, str) else "" for m in system_msgs
        )

        if not combined:
            return None

        block: dict[str, Any] = {"type": "text", "text": combined}

        if self.enable_prompt_caching:
            block["cache_control"] = {"type": "ephemeral"}

        return [block]

    async def _complete_chat_request(
        self, request: ChatRequest, **kwargs
    ) -> ChatResponse:

        self._set_session_from_request(request)
        request.messages = self._get_recent_messages(request.messages)

        logger.debug(
            f"Received ChatRequest with {len(request.messages)} messages (raw={self.raw})"
        )

        system_msgs = [m for m in request.messages if m.role == "system"]
        developer_msgs = [m for m in request.messages if m.role == "developer"]
        conversation = [
            m for m in request.messages if m.role in ("user", "assistant", "tool")
        ]

        system_blocks = self._format_system_with_cache(system_msgs)

        if system_blocks:
            logger.info(
                f"[PROVIDER] System message length: {len(system_blocks[0]['text'])} chars (caching={'cache_control' in system_blocks[0]})"
            )
        else:
            logger.info("[PROVIDER] No system messages")

        context_user_msgs = []
        for i, dev_msg in enumerate(developer_msgs):
            content = dev_msg.content if isinstance(dev_msg.content, str) else ""
            content_preview = content[:100] + ("..." if len(content) > 100 else "")
            logger.info(
                f"[PROVIDER] Converting developer message {i + 1}/{len(developer_msgs)}: length={len(content)}"
            )
            logger.debug(f"[PROVIDER] Developer message preview: {content_preview}")
            wrapped = f"<context_file>\n{content}\n</context_file>"
            context_user_msgs.append({"role": "user", "content": wrapped})

        conversation_msgs = self._convert_messages(
            [m.model_dump() for m in conversation]
        )
        all_messages = context_user_msgs + conversation_msgs
        all_messages = self._apply_message_cache_control(all_messages)

        params = {
            "model": kwargs.get("model", self.default_model),
            "messages": all_messages,
            "max_tokens": request.max_output_tokens
            or kwargs.get("max_tokens", self.max_tokens),
            # NOTE: `request.temperature or ...` would silently fall through
            # to the default when an explicit temperature=0.0 is requested
            # (0.0 is falsy). Use explicit None check instead.
            "temperature": (
                request.temperature
                if request.temperature is not None
                else kwargs.get("temperature", self.temperature)
            ),
        }

        if system_blocks:
            params["system"] = system_blocks

        self._available_tool_names = (
            {tool.name for tool in request.tools} if request.tools else set()
        )
        if request.tools:
            tools = self._convert_tools_from_request(request.tools)

            # Store definitions for list_tools retrieval (before injecting list_tools itself)
            self._last_tools_text = self._format_tools_listing(tools)

            # Inject provider-internal list_tools so the model can recover
            # tool definitions after CLI-side context compaction.
            tools.append(
                {
                    "name": "list_tools",
                    "description": (
                        "Returns the full definitions of all available tools. "
                        "Call this if tool definitions are no longer visible "
                        "after context compaction."
                    ),
                    "input_schema": {"type": "object", "properties": {}},
                }
            )
            self._available_tool_names.add("list_tools")

            params["tools"] = self._apply_tool_cache_control(tools)
            if tool_choice := kwargs.get("tool_choice"):
                params["tool_choice"] = tool_choice

        web_search_enabled = kwargs.get("enable_web_search", self.enable_web_search)
        if web_search_enabled:
            web_search_tool = self._build_web_search_tool(kwargs)
            if "tools" not in params:
                params["tools"] = []
            params["tools"].insert(0, web_search_tool)

        thinking_enabled = bool(kwargs.get("extended_thinking"))
        reasoning_effort = getattr(request, "reasoning_effort", None)
        if "extended_thinking" not in kwargs and reasoning_effort is not None:
            thinking_enabled = True

        thinking_budget = None
        interleaved_thinking_enabled = False
        request_caps = self._get_capabilities(params["model"])
        if thinking_enabled:
            if not request_caps.supports_thinking:
                logger.info(
                    "[PROVIDER] Model %s does not support extended thinking"
                    " -- ignoring thinking request",
                    params["model"],
                )
                thinking_enabled = False
            else:
                effort_budget: int | None = None
                if reasoning_effort == "low":
                    effort_budget = 4096
                elif reasoning_effort in ("medium", "high"):
                    effort_budget = request_caps.default_thinking_budget

                budget_tokens = (
                    kwargs.get("thinking_budget_tokens")
                    or effort_budget
                    or self.config.get("thinking_budget_tokens")
                    or request_caps.default_thinking_budget
                )
                buffer_tokens = kwargs.get("thinking_budget_buffer") or self.config.get(
                    "thinking_budget_buffer", 4096
                )

                thinking_budget = budget_tokens
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }

                params["temperature"] = 1.0

                model_ceiling = request_caps.max_output_tokens
                target_tokens = min(budget_tokens + buffer_tokens, model_ceiling)
                if params.get("max_tokens"):
                    params["max_tokens"] = min(
                        max(params["max_tokens"], target_tokens), model_ceiling
                    )
                else:
                    params["max_tokens"] = target_tokens

                interleaved_thinking_enabled = False

        if stop_sequences := kwargs.get("stop_sequences"):
            params["stop_sequences"] = stop_sequences

        if self.coordinator and hasattr(self.coordinator, "hooks"):
            request_payload: dict[str, Any] = {
                "provider": "anthropic",
                "model": params["model"],
                "message_count": len(params["messages"]),
                "has_system": bool(system_blocks),
                "thinking_enabled": thinking_enabled,
                "thinking_budget": thinking_budget,
                "interleaved_thinking": interleaved_thinking_enabled,
            }
            if self.raw:
                request_payload["raw"] = redact_secrets(params)
            await self.coordinator.hooks.emit("llm:request", request_payload)

        start_time = time.time()
        model = params["model"]

        async def _do_complete() -> ParsedMessage:
            _client_ref: ClaudeSDKClient | None = None
            try:
                async with asyncio.timeout(self.timeout):
                    async with ClaudeSDKClient(
                        options=ClaudeAgentOptions(
                            tools=[],  # disable built-in tools
                            model=self.default_model,
                            resume=self._session.id,
                            system_prompt="---",  # system prompt is passed in messages
                            max_thinking_tokens=self.max_thinking_tokens,
                            betas=self._beta_headers,
                        )
                    ) as client:
                        _client_ref = client
                        prompt = self._convert_prompt_from_request_params(params)
                        await client.query(prompt, session_id=self._session.id)
                        result = await self._parse_response(client, model)

                        try:
                            transport = client._transport
                            await transport.end_input()
                            if hasattr(transport, "_process") and transport._process:
                                await asyncio.wait_for(
                                    transport._process.wait(), timeout=5.0
                                )
                        except (asyncio.TimeoutError, Exception):
                            pass

                        return result

            except TimeoutError:
                self._force_kill_subprocess(_client_ref)
                raise KernelLLMTimeoutError(
                    f"Request timed out after {self.timeout}s",
                    provider="claude",
                    model=model,
                    retryable=True,
                ) from None

            except asyncio.CancelledError:
                self._force_kill_subprocess(_client_ref)
                raise

            except KernelLLMError:
                self._force_kill_subprocess(_client_ref)
                raise

            except Exception as e:
                self._force_kill_subprocess(_client_ref)
                body = getattr(e, "body", None)
                error_msg = json.dumps(body) if body is not None else str(e)
                if not error_msg:
                    error_msg = f"{type(e).__name__}: (no message)"
                raise KernelLLMError(
                    error_msg,
                    provider="claude",
                    model=model,
                    retryable=True,
                ) from e

        async def _on_retry(attempt: int, delay: float, error: KernelLLMError) -> None:
            logger.warning(
                "[PROVIDER] Retry %d/%d in %.1fs: %s",
                attempt,
                self._retry_config.max_retries,
                delay,
                error,
            )
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "provider:retry",
                    {
                        "provider": "claude",
                        "model": model,
                        "attempt": attempt,
                        "max_retries": self._retry_config.max_retries,
                        "delay": delay,
                        "error": str(error),
                        "retry_after": getattr(error, "retry_after", None),
                    },
                )

        try:
            response = await retry_with_backoff(
                _do_complete, self._retry_config, on_retry=_on_retry
            )
        except (KernelLLMError, asyncio.CancelledError):
            elapsed_ms = int((time.time() - start_time) * 1000)
            if self.coordinator and hasattr(self.coordinator, "hooks"):
                await self.coordinator.hooks.emit(
                    "llm:response",
                    {
                        "provider": "anthropic",
                        "model": model,
                        "status": "error",
                        "duration_ms": elapsed_ms,
                    },
                )
            raise

        elapsed_ms = int((time.time() - start_time) * 1000)

        if self.coordinator and hasattr(self.coordinator, "hooks"):
            response_event: dict[str, Any] = {
                "provider": "anthropic",
                "model": params["model"],
                "usage": {
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                    **(
                        {"cache_read": response.usage.cache_read_input_tokens}
                        if hasattr(response.usage, "cache_read_input_tokens")
                        and response.usage.cache_read_input_tokens
                        else {}
                    ),
                    **(
                        {"cache_write": response.usage.cache_creation_input_tokens}
                        if hasattr(response.usage, "cache_creation_input_tokens")
                        and response.usage.cache_creation_input_tokens
                        else {}
                    ),
                },
                "status": "ok",
                "duration_ms": elapsed_ms,
            }

            if self.raw:
                response_event["raw"] = redact_secrets(response.model_dump())
            await self.coordinator.hooks.emit("llm:response", response_event)

        return self._convert_to_chat_response(response)

    def _force_kill_subprocess(self, client: "ClaudeSDKClient | None") -> None:
        if client is None:
            return
        try:
            transport = client._transport
            if hasattr(transport, "_process") and transport._process:
                if transport._process.returncode is None:
                    logger.warning(
                        "[PROVIDER] Force-killing CLI subprocess (PID: %s)",
                        transport._process.pid,
                    )
                    transport._process.kill()
        except Exception as exc:
            logger.debug(f"[PROVIDER] Error killing subprocess: {exc}")

    def parse_tool_calls(self, response: ChatResponse) -> list[ToolCall]:
        if not response.tool_calls:
            return []

        valid_calls = []
        for tc in response.tool_calls:
            if tc.arguments is None:
                logger.debug(f"Filtering out tool '{tc.name}' with missing arguments")
                continue
            valid_calls.append(tc)

        if len(valid_calls) < len(response.tool_calls):
            logger.info(
                f"Filtered {len(response.tool_calls) - len(valid_calls)} tool calls with empty arguments"
            )

        return valid_calls

    def _clean_content_block(self, block: dict[str, Any]) -> dict[str, Any]:
        """Remove fields not accepted by Anthropic API from a content block."""
        block_type = block.get("type")

        if block_type == "text":
            return {"type": "text", "text": block.get("text", "")}
        if block_type == "thinking":
            cleaned = {"type": "thinking", "thinking": block.get("thinking", "")}
            if "signature" in block:
                cleaned["signature"] = block["signature"]
            return cleaned
        if block_type == "tool_use" or block_type == "tool_call":
            return {
                "type": "tool_use",
                "id": block.get("id", ""),
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            }
        if block_type == "tool_result":
            return {
                "type": "tool_result",
                "tool_use_id": block.get("tool_use_id", ""),
                "content": block.get("content", ""),
            }
        if block_type == "web_search_tool_result":
            cleaned: dict[str, Any] = {
                "type": "web_search_tool_result",
            }
            if "tool_use_id" in block:
                cleaned["tool_use_id"] = block["tool_use_id"]
            if "content" in block:
                cleaned["content"] = block["content"]
            return cleaned
        cleaned = dict(block)
        cleaned.pop("visibility", None)
        return cleaned

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format, batching tool results."""
        valid_tool_use_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                if msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        tc_id = tc.get("id") or tc.get("tool_call_id")
                        if tc_id:
                            valid_tool_use_ids.add(tc_id)
                content = msg.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") in (
                            "tool_call",
                            "tool_use",
                        ):
                            tc_id = block.get("id")
                            if tc_id:
                                valid_tool_use_ids.add(tc_id)

        anthropic_messages = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                i += 1
                continue

            if role == "tool":
                tool_results = []
                skipped_count = 0
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_use_id = tool_msg.get("tool_call_id")

                    if not tool_use_id or tool_use_id not in valid_tool_use_ids:
                        logger.warning(
                            f"Skipping orphaned tool_result (no matching tool_use): "
                            f"tool_call_id={tool_use_id}, content_preview={str(tool_msg.get('content', ''))[:100]}"
                        )
                        skipped_count += 1
                        i += 1
                        continue

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": tool_msg.get("content", ""),
                        }
                    )
                    i += 1

                if tool_results:
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": tool_results,
                        }
                    )
                elif skipped_count > 0:
                    logger.warning(
                        f"All {skipped_count} consecutive tool_results were orphaned and skipped"
                    )
                continue  # i already advanced in while loop
            if role == "assistant":
                if "tool_calls" in msg and msg["tool_calls"]:
                    content_blocks = []

                    has_thinking = "thinking_block" in msg and msg["thinking_block"]
                    if has_thinking:
                        cleaned_thinking = self._clean_content_block(
                            msg["thinking_block"]
                        )
                        content_blocks.append(cleaned_thinking)

                    if content and not has_thinking:
                        if isinstance(content, list):
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    content_blocks.append(
                                        {"type": "text", "text": block.get("text", "")}
                                    )
                                elif (
                                    not isinstance(block, dict)
                                    and hasattr(block, "type")
                                    and block.type == "text"
                                ):
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "text": getattr(block, "text", ""),
                                        }
                                    )
                        else:
                            content_blocks.append({"type": "text", "text": content})

                    for tc in msg["tool_calls"]:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("tool", ""),
                                "input": tc.get("arguments", {}),
                            }
                        )

                    anthropic_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                elif "thinking_block" in msg and msg["thinking_block"]:
                    cleaned_thinking = self._clean_content_block(msg["thinking_block"])
                    content_blocks = [cleaned_thinking]
                    if content:
                        if isinstance(content, list):
                            for block in content:
                                if (
                                    isinstance(block, dict)
                                    and block.get("type") == "text"
                                ):
                                    content_blocks.append(
                                        {"type": "text", "text": block.get("text", "")}
                                    )
                                elif (
                                    not isinstance(block, dict)
                                    and hasattr(block, "type")
                                    and block.type == "text"
                                ):
                                    content_blocks.append(
                                        {
                                            "type": "text",
                                            "text": getattr(block, "text", ""),
                                        }
                                    )
                        else:
                            content_blocks.append({"type": "text", "text": content})
                    anthropic_messages.append(
                        {"role": "assistant", "content": content_blocks}
                    )
                else:
                    if isinstance(content, list):
                        cleaned_blocks = [
                            self._clean_content_block(block) for block in content
                        ]
                        anthropic_messages.append(
                            {"role": "assistant", "content": cleaned_blocks}
                        )
                    else:
                        anthropic_messages.append(
                            {"role": "assistant", "content": content}
                        )
                i += 1
            elif role == "developer":
                wrapped = f"<context_file>\n{content}\n</context_file>"
                anthropic_messages.append({"role": "user", "content": wrapped})
                i += 1
            else:
                if isinstance(content, list):
                    content_blocks = []
                    for block in content:
                        if isinstance(block, dict):
                            block_type = block.get("type")
                            if block_type == "text":
                                content_blocks.append(
                                    {"type": "text", "text": block.get("text", "")}
                                )
                            elif block_type == "image":
                                source = block.get("source", {})
                                if source.get("type") == "base64":
                                    content_blocks.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": source.get(
                                                    "media_type", "image/jpeg"
                                                ),
                                                "data": source.get("data"),
                                            },
                                        }
                                    )
                                else:
                                    logger.warning(
                                        f"Unsupported image source type: {source.get('type')}"
                                    )

                    if content_blocks:
                        anthropic_messages.append(
                            {"role": "user", "content": content_blocks}
                        )
                else:
                    anthropic_messages.append({"role": "user", "content": content})
                i += 1

        return anthropic_messages

    def _convert_tools_from_request(self, tools: list) -> list[dict[str, Any]]:
        anthropic_tools = []
        for tool in tools:
            tool_type = getattr(tool, "type", None)
            if tool_type and tool_type != "function":
                if hasattr(tool, "model_dump"):
                    anthropic_tools.append(tool.model_dump(exclude_none=True))
                elif isinstance(tool, dict):
                    anthropic_tools.append(tool)
                else:
                    native_tool: dict[str, Any] = {"type": tool_type}
                    if hasattr(tool, "name") and tool.name:
                        native_tool["name"] = tool.name
                    if hasattr(tool, "max_uses") and tool.max_uses is not None:
                        native_tool["max_uses"] = tool.max_uses
                    if (
                        hasattr(tool, "user_location")
                        and tool.user_location is not None
                    ):
                        native_tool["user_location"] = tool.user_location
                    anthropic_tools.append(native_tool)
                logger.debug(f"[PROVIDER] Added native tool: {tool_type}")
            else:
                anthropic_tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": tool.parameters,
                    }
                )
        return anthropic_tools

    def _extract_web_search_citations(self, block: Any) -> list[dict[str, Any]]:
        citations = []
        content = getattr(block, "content", None)
        if not content:
            return citations

        results = content if isinstance(content, list) else [content]

        for result in results:
            if hasattr(result, "type") and result.type == "web_search_result":
                citation: dict[str, Any] = {}

                if hasattr(result, "url") and result.url:
                    citation["url"] = result.url
                elif hasattr(result, "source_url") and result.source_url:
                    citation["url"] = result.source_url

                if hasattr(result, "title") and result.title:
                    citation["title"] = result.title

                if hasattr(result, "snippet") and result.snippet:
                    citation["snippet"] = result.snippet
                elif hasattr(result, "description") and result.description:
                    citation["snippet"] = result.description
                elif hasattr(result, "encrypted_content") and result.encrypted_content:
                    citation["has_content"] = True

                if citation.get("url"):
                    citations.append(citation)

        return citations

    def _build_web_search_tool(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        tool: dict[str, Any] = {
            "type": "web_search_20250305",
            "name": "web_search",
        }

        max_uses = kwargs.get("web_search_max_uses") or self.config.get(
            "web_search_max_uses"
        )
        if max_uses is not None:
            tool["max_uses"] = max_uses

        user_location = kwargs.get("web_search_user_location") or self.config.get(
            "web_search_user_location"
        )
        if user_location is not None:
            tool["user_location"] = user_location

        return tool

    def _apply_tool_cache_control(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not tools or not self.enable_prompt_caching:
            return tools

        tools[-1]["cache_control"] = {"type": "ephemeral"}
        return tools

    def _apply_message_cache_control(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not messages or not self.enable_prompt_caching:
            return messages

        last_msg = messages[-1]
        content = last_msg.get("content")

        if isinstance(content, list) and content:
            content[-1]["cache_control"] = {"type": "ephemeral"}
        elif isinstance(content, str):
            last_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        return messages

    def _convert_to_chat_response(self, response: ParsedMessage) -> ChatResponse:

        content_blocks = []
        tool_calls = []
        web_search_results: list[dict[str, Any]] = []
        event_blocks: list[
            TextContent | ThinkingContent | ToolCallContent | WebSearchContent | Session
        ] = []
        text_accumulator: list[str] = []

        for block in response.content:
            if block.type == "text":
                content_blocks.append(TextBlock(text=block.text))
                text_accumulator.append(block.text)
                event_blocks.append(TextContent(text=block.text))
            elif block.type == "thinking":
                content_blocks.append(
                    ThinkingBlock(
                        thinking=block.thinking,
                        signature=getattr(block, "signature", None),
                        visibility="internal",
                    )
                )
                event_blocks.append(ThinkingContent(text=block.thinking))
            elif block.type == "tool_use" or block.type == "tool_call":
                content_blocks.append(
                    ToolCallBlock(id=block.id, name=block.name, input=block.input)
                )
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )
                event_blocks.append(
                    ToolCallContent(id=block.id, name=block.name, arguments=block.input)
                )
            elif block.type == "web_search_tool_result":
                citations = self._extract_web_search_citations(block)
                web_search_results.append(
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": getattr(block, "tool_use_id", None),
                        "citations": citations,
                    }
                )
                event_blocks.append(
                    WebSearchContent(
                        query=getattr(block, "query", ""),
                        citations=citations,
                    )
                )
        cache_creation = getattr(response.usage, "cache_creation_input_tokens", None)
        cache_read = getattr(response.usage, "cache_read_input_tokens", None)

        # Gross input_tokens must include cache-read tokens. The Anthropic API
        # reports `input_tokens` exclusive of cache reads, but cache reads are
        # billable and should appear in the total. Cache-creation tokens are
        # already counted in `input_tokens` by the API.
        gross_input_tokens = response.usage.input_tokens + (cache_read or 0)
        usage_kwargs: dict[str, Any] = {
            "input_tokens": gross_input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": gross_input_tokens + response.usage.output_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_creation,
        }

        if cache_creation is not None:
            usage_kwargs["cache_creation_input_tokens"] = cache_creation
        if cache_read is not None:
            usage_kwargs["cache_read_input_tokens"] = cache_read

        usage = Usage(**usage_kwargs)
        combined_text = "\n\n".join(text_accumulator).strip()
        content_blocks.append(self._session)

        return ClaudeChatResponse(
            content=content_blocks,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            finish_reason=response.stop_reason,
            content_blocks=event_blocks if event_blocks else None,
            text=combined_text or None,
            web_search_results=web_search_results if web_search_results else None,
        )

    def _convert_prompt_from_request_params(
        self, params: dict[str, list[dict[str, str | list[dict[str, Any]]]]]
    ) -> str:
        system: list[str] = []
        tools: list[str] = []
        system_reminders: list[str] = []
        user_messages: list[str] = []
        tool_results: list[str] = []

        for message in params.get("system", []):
            system.append(message["text"])

        for message in params.get("tools", []):
            tools.append(
                f"""{{"name": "{message["name"]}", "input_schema": {json.dumps(message["input_schema"])}}}\n<instructions>\n{message["description"]}\n</instructions>"""
            )

        for message in params.get("messages", []):
            match message["role"]:
                case "user":
                    match message["content"]:
                        case str():
                            user_messages.append(f"[user]: {message['content']}")

                        case list():
                            for block in message["content"]:
                                match block["type"]:
                                    case "tool_result":
                                        try:
                                            output = json.loads(block["content"])
                                        except json.JSONDecodeError:
                                            output = block["content"]

                                        tool_json = json.dumps(
                                            {
                                                "id": block["tool_use_id"],
                                                "output": output,
                                            }
                                        )
                                        tool_results.append(f"[tool]: {tool_json}")
                                    case "text":
                                        system_reminders.append(f"""{block["text"]}""")

                                    case _:
                                        raise NotImplementedError(
                                            f"[PROVIDER] Unknown block type for user message: {block['type']}"
                                        )
                        case _:
                            raise NotImplementedError(
                                f"[PROVIDER] Unknown content type for user message: {type(message['content'])}"
                            )

                case "assistant":
                    match message["content"]:
                        case list():
                            for block in message["content"]:
                                match block["type"]:
                                    case (
                                        "thinking"
                                        | "tool_use"
                                        | "tool_call"
                                        | "text"
                                        | "redacted_thinking"
                                    ):
                                        pass

                                    case _:
                                        raise NotImplementedError(
                                            f"[PROVIDER] Unknown block type for assistant message: {block['type']}"
                                        )
                        case _:
                            raise NotImplementedError(
                                f"[PROVIDER] Unknown content type for assistant message: {type(message['content'])}"
                            )
                case _:
                    raise NotImplementedError(
                        f"[PROVIDER] Unknown message role: {message['role']}"
                    )

        prompt = ""

        if not self._session.id:
            prompt += "<system>\n" + "\n\n".join(system) + "\n</system>\n\n"
            prompt += "<tools>\n" + "\n\n".join(tools) + "\n</tools>\n\n"

        if tool_results:
            prompt += "\n".join(tool_results) + "\n\n"

        if user_messages:
            prompt += "\n".join(user_messages) + "\n\n"

        if system_reminders:
            prompt += "\n".join(system_reminders) + "\n\n"

        # Always include tool use reminder — do not gate on system_reminders.
        # CLI-side compaction or missing hook injections can leave system_reminders
        # empty, causing the model to fall back to <tool_call> XML hallucinations.
        prompt += f"""<system-reminder source="hooks-tools-reminder">\n{TOOL_USE_REMINDER}</system-reminder>"""

        return prompt

    def _parse_tool_blocks_from_text(
        self, text: str
    ) -> tuple[list[AnthropicToolUseBlock], str]:

        tool_blocks: list[AnthropicToolUseBlock] = []
        spans_to_remove: list[tuple[int, int]] = []

        tokenizer = re.compile(r"(```(?:.|\n)*?```)|(`[^`\n]*`)|(\[tool\]:)", re.DOTALL)

        pos = 0
        while pos < len(text):
            match = tokenizer.search(text, pos)
            if not match:
                break

            start, end = match.span()
            if match.group(1):
                pos = end

            elif match.group(2):
                pos = end

            elif match.group(3):
                json_start = end

                while json_start < len(text) and text[json_start].isspace():
                    json_start += 1

                try:
                    decoder = json.JSONDecoder()
                    obj, end_offset = decoder.raw_decode(text, idx=json_start)

                    tool_blocks.append(AnthropicToolUseBlock.model_validate(obj))
                    spans_to_remove.append((start, end_offset))
                    pos = end_offset

                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(
                        "[PROVIDER] Failed to parse tool block from text: %s", e
                    )
                    pos = end

        if self._available_tool_names and tool_blocks:
            validated = []
            for block in tool_blocks:
                if block.name in self._available_tool_names:
                    validated.append(block)
                else:
                    logger.warning(
                        "[PROVIDER] Filtered tool block with unknown name '%s' "
                        "(id=%s). Not in %d available tools. "
                        "Likely an echoed example or hallucination.",
                        block.name,
                        block.id,
                        len(self._available_tool_names),
                    )
            tool_blocks = validated

        if spans_to_remove:
            parts: list[str] = []
            prev_end = 0
            for rm_start, rm_end in spans_to_remove:
                parts.append(text[prev_end:rm_start])
                prev_end = rm_end
            parts.append(text[prev_end:])
            cleaned = "".join(parts)
        else:
            cleaned = text

        return tool_blocks, cleaned

    def _classify_and_raise_error(self, error_text: str, model: str) -> None:
        """Classify a CLI error message and raise the appropriate kernel error type."""
        lower = error_text.lower()
        if "rate limit" in lower or "rate_limit" in lower or "429" in lower:
            raise KernelRateLimitError(
                error_text,
                provider="claude",
                model=model,
                retryable=True,
            )
        elif "overloaded" in lower or "529" in lower or "capacity" in lower:
            raise KernelProviderUnavailableError(
                error_text,
                provider="claude",
                model=model,
                retryable=True,
                delay_multiplier=self._overloaded_delay_multiplier,
            )
        else:
            raise KernelLLMError(
                error_text,
                provider="claude",
                model=model,
                retryable=True,
            )

    async def _parse_response(
        self, client: ClaudeSDKClient, model: str
    ) -> ParsedMessage:

        response_model: str = ""
        content: list[ParsedContentBlock] = []
        usage = AnthropicUsage(input_tokens=0, output_tokens=0)
        received_result = False
        error_result: str | None = None

        async for message in client.receive_response():
            match message:
                case claude_agent_sdk.types.AssistantMessage():
                    response_model = message.model
                    for block in message.content:
                        match block:
                            case claude_agent_sdk.types.TextBlock():
                                tool_blocks, cleaned_text = (
                                    self._parse_tool_blocks_from_text(block.text)
                                )
                                if tool_blocks:
                                    content.extend(tool_blocks)
                                if (
                                    cleaned_text.strip()
                                    and cleaned_text != "(no content)"
                                ):
                                    content.append(
                                        ParsedTextBlock(
                                            type="text",
                                            text=cleaned_text,
                                        )
                                    )

                            case claude_agent_sdk.types.ThinkingBlock():
                                content.append(
                                    AnthropicThinkingBlock(
                                        type="thinking",
                                        thinking=block.thinking,
                                        signature=block.signature,
                                    )
                                )
                            case claude_agent_sdk.types.ToolUseBlock():
                                content.append(
                                    AnthropicToolUseBlock(
                                        type="tool_use",
                                        id=block.id,
                                        name=block.name,
                                        input=block.input,
                                    )
                                )
                            case _:
                                logger.debug(
                                    "[PROVIDER] AssistantMessage content block type ignored: %s",
                                    type(block),
                                )

                case claude_agent_sdk.types.ResultMessage():
                    received_result = True
                    self._session.id = message.session_id
                    if message.usage:
                        usage.input_tokens = message.usage.get(
                            "input_tokens",
                            0,
                        )
                        usage.output_tokens = message.usage.get(
                            "output_tokens",
                            0,
                        )
                        usage.cache_read_input_tokens = message.usage.get(
                            "cache_read_input_tokens",
                            None,
                        )
                        usage.cache_creation_input_tokens = message.usage.get(
                            "cache_creation_input_tokens",
                            None,
                        )

                    if message.is_error:
                        error_result = str(message.result or "")
                        logger.warning(
                            "[PROVIDER] SDK response indicates error: %s",
                            error_result,
                        )

                case (
                    claude_agent_sdk.types.SystemMessage()
                    | claude_agent_sdk.types.UserMessage()
                    | claude_agent_sdk.types.StreamEvent()
                ):
                    logger.debug(
                        f"[PROVIDER] SDK message type ignored: {type(message)}"
                    )

                case _:
                    logger.debug(
                        f"[PROVIDER] SDK message type ignored: {type(message)}"
                    )

        if not received_result:
            raise KernelLLMError(
                "[PROVIDER] CLI subprocess ended without delivering a ResultMessage. "
                "The subprocess may have crashed or its stdout pipe was blocked.",
                provider="claude",
                model=model,
                retryable=True,
            )

        if error_result:
            self._classify_and_raise_error(error_result, model)

        stop_reason = (
            "tool_use"
            if any(isinstance(block, AnthropicToolUseBlock) for block in content)
            else "end_turn"
        )

        return ParsedMessage(
            id=self._session.id,
            content=content,
            model=response_model or model,
            role="assistant",
            type="message",
            usage=usage,
            stop_reason=stop_reason,
        )

    def _set_session_from_request(self, request: ChatRequest):
        for message in reversed(request.messages):
            if message.role == "assistant" and isinstance(message.content, list):
                for block in message.content:
                    if (
                        hasattr(block, "type")
                        and block.type == "redacted_thinking"
                        and hasattr(block, "data")
                        and block.data.startswith(SESSION_TAG)
                    ):
                        self._session.data = block.data
                        return

        self._session = Session()


TOOL_USE_EXAMPLE1 = json.dumps(
    {
        "type": "tool_use",
        "name": "first_tool",
        "id": "tl4xcu5",
        "input": {"param1": "value1", "param2": 42},
    }
)

TOOL_USE_EXAMPLE2 = json.dumps(
    {
        "type": "tool_use",
        "name": "second_tool",
        "id": "tl4t214",
        "input": {"param1": "valueX"},
    }
)

TOOL_USE_REMINDER = f"""You have access to the all the tools defined with the <tools> XML block.
To call tools respond with tool blocks with a valid JSON with "name", "id", and "input" fields.
In the example below two tools are being called in parallel.
<example>
[tool]: {TOOL_USE_EXAMPLE1}
[tool]: {TOOL_USE_EXAMPLE2}
</example>
<instructions>
Usage:
- The response must ONLY contain tool blocks. No additional text.
- Generate a 7 character high-entropy id for each tool block
- The "input" field must respect the "input_schema" in the tool definitions
- Wait for the next turn for the tool results
- If tool definitions are no longer visible after context compaction, call the "list_tools" tool to retrieve them.
</instructions>
"""

INTERLEAVED_THINKING_REMINDER = """Before proceeding, briefly reflect on what you just learned from the tool result. Think through this internally, then continue with your work."""
