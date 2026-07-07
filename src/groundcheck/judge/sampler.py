"""Judge abstraction: MCP sampling (zero-key) with an optional direct-API fallback."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol

import httpx
from mcp import types

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


class Judge(Protocol):
    """Anything that can turn (system, prompt) into a model completion."""

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str: ...


class NoJudgeAvailable(RuntimeError):
    """Raised when neither MCP sampling nor an API key fallback is available."""

    def __init__(self) -> None:
        super().__init__(
            "This tool needs an LLM to judge semantic content, and none is available. "
            "Either (1) connect via an MCP client that supports sampling (Claude Desktop, "
            "Claude Code, and most modern MCP hosts do), or (2) set the ANTHROPIC_API_KEY "
            "environment variable to fall back to a direct Anthropic API call."
        )


class SamplingJudge:
    """Zero-key judge: asks the connected MCP client's own LLM via sampling."""

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        result = await self._ctx.session.create_message(
            messages=[
                types.SamplingMessage(
                    role="user", content=types.TextContent(type="text", text=prompt)
                )
            ],
            max_tokens=max_tokens,
            system_prompt=system,
        )
        if isinstance(result.content, types.TextContent):
            return result.content.text
        raise ValueError(f"Client returned non-text sampling content: {type(result.content)!r}")


class ApiKeyJudge:
    """Fallback judge: calls the Anthropic Messages API directly with ANTHROPIC_API_KEY.

    Used only when the connected MCP client doesn't declare the `sampling`
    capability. Never required for the happy path.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL) -> None:
        self._api_key = api_key
        self._model = model

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            data = response.json()
            return "".join(block["text"] for block in data["content"] if block["type"] == "text")


def api_key_judge_from_env() -> ApiKeyJudge | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
    return ApiKeyJudge(api_key, model=model)


def client_supports_sampling(ctx: Context) -> bool:
    return ctx.session.check_client_capability(
        types.ClientCapabilities(sampling=types.SamplingCapability())
    )


def get_judge(ctx: Context) -> Judge:
    """Prefer zero-key MCP sampling; fall back to ANTHROPIC_API_KEY; else raise
    NoJudgeAvailable with an actionable message naming both options."""
    if client_supports_sampling(ctx):
        return SamplingJudge(ctx)
    fallback = api_key_judge_from_env()
    if fallback is not None:
        return fallback
    raise NoJudgeAvailable()
