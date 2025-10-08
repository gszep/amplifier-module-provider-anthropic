"""
Anthropic provider module for Amplifier.
Integrates with Anthropic's Claude API.
"""

import logging
import os
from typing import Any
from typing import Optional

from amplifier_core import ModuleCoordinator
from amplifier_core import ProviderResponse
from amplifier_core import ToolCall
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


async def mount(coordinator: ModuleCoordinator, config: dict[str, Any] | None = None):
    """
    Mount the Anthropic provider.

    Args:
        coordinator: Module coordinator
        config: Provider configuration including API key

    Returns:
        Optional cleanup function
    """
    config = config or {}

    # Get API key from config or environment
    api_key = config.get("api_key")
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        logger.warning("No API key found for Anthropic provider")
        return None

    provider = AnthropicProvider(api_key, config)
    await coordinator.mount("providers", provider, name="anthropic")
    logger.info("Mounted AnthropicProvider")

    # Return cleanup function
    async def cleanup():
        if hasattr(provider.client, "close"):
            await provider.client.close()

    return cleanup


class AnthropicProvider:
    """Anthropic API integration."""

    name = "anthropic"

    def __init__(self, api_key: str, config: dict[str, Any] | None = None):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key
            config: Additional configuration
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.config = config or {}
        self.default_model = self.config.get("default_model", "claude-3-5-sonnet-20241022")
        self.max_tokens = self.config.get("max_tokens", 4096)
        self.temperature = self.config.get("temperature", 0.7)

    async def complete(self, messages: list[dict[str, Any]], **kwargs) -> ProviderResponse:
        """
        Generate completion from messages.

        Args:
            messages: Conversation history
            **kwargs: Additional parameters

        Returns:
            Provider response
        """
        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages(messages)

        # Extract system message if present
        system = None
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
                break

        # Prepare request parameters
        params = {
            "model": kwargs.get("model", self.default_model),
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if system:
            params["system"] = system

        # Add tools if provided
        if "tools" in kwargs:
            params["tools"] = self._convert_tools(kwargs["tools"])

        try:
            response = await self.client.messages.create(**params)

            # Convert response to standard format
            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(tool=block.name, arguments=block.input, id=block.id))

            return ProviderResponse(
                content=content,
                raw=response,
                usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                tool_calls=tool_calls if tool_calls else None,
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def parse_tool_calls(self, response: ProviderResponse) -> list[ToolCall]:
        """
        Parse tool calls from provider response.

        Args:
            response: Provider response

        Returns:
            List of tool calls
        """
        return response.tool_calls or []

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert messages to Anthropic format."""
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            # Skip system messages (handled separately)
            if role == "system":
                continue

            # Convert role names
            if role == "tool":
                # Tool results in Anthropic format
                tool_use_id = msg.get("tool_call_id")
                if not tool_use_id:
                    logger.warning(f"Tool result missing tool_call_id: {msg}")
                    tool_use_id = "unknown"  # Fallback, but will likely fail

                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": content,
                            }
                        ],
                    }
                )
            elif role == "assistant":
                # Assistant messages - check for tool calls
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Assistant message with tool calls
                    content_blocks = []

                    # Add text content if present
                    if content:
                        content_blocks.append({"type": "text", "text": content})

                    # Add tool_use blocks
                    for tc in msg["tool_calls"]:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": tc.get("tool", ""),
                                "input": tc.get("arguments", {}),
                            }
                        )

                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    # Regular assistant message
                    anthropic_messages.append({"role": "assistant", "content": content})
            else:
                # User messages
                anthropic_messages.append({"role": "user", "content": content})

        return anthropic_messages

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format."""
        anthropic_tools = []

        for tool in tools:
            # Get schema from tool if available, otherwise use empty schema
            input_schema = getattr(tool, "input_schema", {"type": "object", "properties": {}, "required": []})

            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": input_schema,
                }
            )

        return anthropic_tools
