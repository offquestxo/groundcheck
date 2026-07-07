"""Test doubles. No real LLM calls happen in unit tests -- FakeJudge returns canned text."""

from __future__ import annotations


class FakeJudge:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls.append({"system": system, "prompt": prompt, "max_tokens": max_tokens})
        return self._responses.pop(0)
