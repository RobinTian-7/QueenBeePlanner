"""``OpenAIChatClient`` request construction, offline (a fake ``openai`` module
records each request): per-model thinking switches, provider defaults and
the bounded-client options."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from exp_graph.llm import factory
from exp_graph.llm.openai_client import OpenAIChatClient


_DEFAULT_USAGE = object()


def _request_for_model(
    monkeypatch,
    model_name: str,
    response_usage=_DEFAULT_USAGE,
    **client_kwargs,
) -> dict[str, object]:
    request: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **kwargs):
            request.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))
                ],
                usage=(
                    SimpleNamespace(prompt_tokens=3, completion_tokens=2)
                    if response_usage is _DEFAULT_USAGE
                    else response_usage
                ),
            )

    class FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            request["__init__"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    openai_module = ModuleType("openai")
    openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_module)
    for name in ("OPENAI_REASONING_EFFORT", "OPENAI_MAX_COMPLETION_TOKENS"):
        monkeypatch.delenv(name, raising=False)

    client = OpenAIChatClient(**client_kwargs)
    client.complete("return json", model_name=model_name)
    return request


def test_json_completion_disables_thinking_mode_for_qwen3_mixed_models(monkeypatch) -> None:
    request = _request_for_model(monkeypatch, "qwen3.5-27b")

    assert request["extra_body"] == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_json_completion_disables_thinking_mode_for_qwen36_model(monkeypatch) -> None:
    request = _request_for_model(monkeypatch, "qwen3.6-27b")

    assert request["extra_body"] == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_json_completion_disables_thinking_mode_for_deepseek_v4_model(monkeypatch) -> None:
    request = _request_for_model(monkeypatch, "deepseek-v4-flash")

    assert request["extra_body"] == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_role_completion_enables_thinking_mode_for_deepseek_model(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    request = _request_for_model(
        monkeypatch,
        "deepseek-v4-flash",
        platform="deepseek",
        thinking_enabled=True,
    )

    assert request["extra_body"] == {"thinking": {"type": "enabled"}}
    assert request["__init__"]["base_url"] == "https://api.deepseek.com"
    assert request["__init__"]["api_key"] == "test-key"


def test_role_completion_disables_thinking_mode_for_deepseek_model(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    request = _request_for_model(
        monkeypatch,
        "deepseek-v4-flash",
        platform="deepseek",
        thinking_enabled=False,
    )

    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_role_completion_disables_thinking_mode_for_bailian_qwen_model(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    request = _request_for_model(
        monkeypatch,
        "qwen3.5-flash",
        platform="bailian",
        thinking_enabled=False,
    )

    assert request["extra_body"] == {
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert request["__init__"]["api_key"] == "test-key"


def test_json_completion_disables_thinking_mode_for_mimo_model(monkeypatch) -> None:
    request = _request_for_model(monkeypatch, "mimo-v2-flash")

    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_role_completion_uses_xiaomi_defaults_for_mimo_model(monkeypatch) -> None:
    monkeypatch.setenv("XIAOMI_API_KEY", "test-key")
    request = _request_for_model(
        monkeypatch,
        "mimo-v2-flash",
        platform="xiaomi",
        thinking_enabled=False,
    )

    assert request["__init__"]["base_url"] == "https://api.xiaomimimo.com/v1"
    assert request["__init__"]["api_key"] == "test-key"
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_factory_accepts_xiaomi_provider(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            seen.update(kwargs)

    monkeypatch.setattr(factory, "OpenAIChatClient", FakeClient)

    factory.create_llm_client("xiaomi", thinking_enabled=False)

    assert seen["platform"] == "xiaomi"
    assert seen["thinking_enabled"] is False


def test_json_completion_disables_thinking_mode_for_kimi_k25_model(monkeypatch) -> None:
    request = _request_for_model(monkeypatch, "kimi-k2.5")

    assert request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert request["temperature"] == 0.6


def test_json_completion_does_not_send_thinking_option_to_non_thinking_model(
    monkeypatch,
) -> None:
    request = _request_for_model(monkeypatch, "deepseek-v3")

    assert "extra_body" not in request


def test_requests_carry_no_extra_headers(monkeypatch) -> None:
    request = _request_for_model(monkeypatch, "gpt-4o-mini")

    assert "extra_headers" not in request
    assert "default_headers" not in request["__init__"]


def test_bounded_client_disables_sdk_retry_and_requires_real_usage(monkeypatch) -> None:
    request = _request_for_model(
        monkeypatch,
        "gpt-4o-mini",
        max_retries=0,
        max_completion_tokens=1024,
        require_provider_usage=True,
    )

    assert request["__init__"]["max_retries"] == 0
    assert request["max_completion_tokens"] == 1024


def test_bounded_client_rejects_missing_provider_usage(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="provider-authoritative"):
        _request_for_model(
            monkeypatch,
            "gpt-4o-mini",
            response_usage=None,
            max_retries=0,
            max_completion_tokens=1024,
            require_provider_usage=True,
        )


@pytest.mark.parametrize(
    ("client_kwargs", "message"),
    [
        ({"max_retries": -1}, "max_retries"),
        ({"max_completion_tokens": 0}, "max_completion_tokens"),
        ({"require_provider_usage": 1}, "require_provider_usage"),
    ],
)
def test_bounded_client_options_reject_invalid_values(
    monkeypatch,
    client_kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _request_for_model(
            monkeypatch,
            "gpt-4o-mini",
            **client_kwargs,
        )


@pytest.mark.parametrize("model_name", ["deepseek-r1", "kimi-k2-thinking"])
def test_json_completion_rejects_thinking_only_model(monkeypatch, model_name: str) -> None:
    with pytest.raises(ValueError, match="thinking-only"):
        _request_for_model(monkeypatch, model_name)
