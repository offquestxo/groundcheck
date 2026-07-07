import httpx
import pytest

from groundcheck.judge.sampler import (
    ApiKeyJudge,
    NoJudgeAvailable,
    SamplingJudge,
    api_key_judge_from_env,
    client_supports_sampling,
    get_judge,
)


class _FakeSession:
    def __init__(self, supports_sampling: bool, response_text: str = "judge said so") -> None:
        self._supports_sampling = supports_sampling
        self._response_text = response_text
        self.create_message_calls: list[dict] = []

    def check_client_capability(self, capability) -> bool:
        return self._supports_sampling

    async def create_message(self, *, messages, max_tokens, system_prompt=None, **kwargs):
        from mcp import types

        self.create_message_calls.append(
            {"messages": messages, "max_tokens": max_tokens, "system_prompt": system_prompt}
        )
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(type="text", text=self._response_text),
            model="fake-model",
        )


class _FakeContext:
    def __init__(self, supports_sampling: bool) -> None:
        self.session = _FakeSession(supports_sampling)


class TestClientSupportsSampling:
    def test_true_when_client_declares_sampling(self):
        assert client_supports_sampling(_FakeContext(supports_sampling=True)) is True

    def test_false_when_client_omits_sampling(self):
        assert client_supports_sampling(_FakeContext(supports_sampling=False)) is False


class TestSamplingJudge:
    async def test_complete_extracts_text(self):
        ctx = _FakeContext(supports_sampling=True)
        judge = SamplingJudge(ctx)
        result = await judge.complete(system="sys", prompt="hello", max_tokens=100)
        assert result == "judge said so"
        assert ctx.session.create_message_calls[0]["system_prompt"] == "sys"
        assert ctx.session.create_message_calls[0]["max_tokens"] == 100


class TestGetJudge:
    def test_prefers_sampling_when_available(self):
        ctx = _FakeContext(supports_sampling=True)
        judge = get_judge(ctx)
        assert isinstance(judge, SamplingJudge)

    def test_falls_back_to_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        ctx = _FakeContext(supports_sampling=False)
        judge = get_judge(ctx)
        assert isinstance(judge, ApiKeyJudge)

    def test_raises_actionable_error_when_neither_available(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        ctx = _FakeContext(supports_sampling=False)
        with pytest.raises(NoJudgeAvailable) as exc_info:
            get_judge(ctx)
        message = str(exc_info.value)
        assert "sampling" in message.lower()
        assert "ANTHROPIC_API_KEY" in message


class TestApiKeyJudgeFromEnv:
    def test_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert api_key_judge_from_env() is None

    def test_builds_judge_when_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
        judge = api_key_judge_from_env()
        assert isinstance(judge, ApiKeyJudge)


class TestApiKeyJudgeComplete:
    async def test_calls_anthropic_api_and_extracts_text(self, monkeypatch):
        captured = {}

        async def fake_post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return httpx.Response(
                200,
                json={"content": [{"type": "text", "text": "the verdict"}]},
                request=httpx.Request("POST", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        judge = ApiKeyJudge("sk-fake", model="claude-sonnet-5")
        result = await judge.complete(system="sys", prompt="prompt", max_tokens=50)
        assert result == "the verdict"
        assert captured["headers"]["x-api-key"] == "sk-fake"
        assert captured["json"]["model"] == "claude-sonnet-5"
