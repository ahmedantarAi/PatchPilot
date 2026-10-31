import asyncio

import pytest

from patchpilot.providers import get_provider
from patchpilot.providers.base import LLMProvider, LLMRequest, Message


def test_mock_provider_smoke() -> None:
    provider = get_provider("mock")
    assert isinstance(provider, LLMProvider)
    req = LLMRequest(messages=[Message(role="user", content="fix this bug")])
    resp = asyncio.run(provider.generate(req))
    assert resp.provider == "mock"
    assert resp.text == "MOCK_RESPONSE"
    assert resp.usage.prompt_tokens == 3


@pytest.mark.parametrize("name", ["ollama", "gemini"])
def test_real_providers_implement_protocol(name: str) -> None:
    provider = get_provider(name)
    assert isinstance(provider, LLMProvider)
    assert provider.name == name
