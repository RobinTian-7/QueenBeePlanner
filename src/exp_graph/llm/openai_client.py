"""OpenAI-backed LLM client for OpenAI-compatible endpoints.

``_DEFAULT_BASE_URLS`` / ``_DEFAULT_API_KEY_ENVS`` hold per-platform presets
of the base URL and of the environment variable that holds the API key
(constructor arguments override them).  The module also maps a reasoning
("thinking") on / off setting to provider-specific request fields and
reports token usage (estimated when the provider reports none).
"""

from __future__ import annotations

import os

from exp_graph.llm.base import LLMResponse, LLMUsage, estimate_tokens


_DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "bailian": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "alibaba": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "xiaomi": "https://api.xiaomimimo.com/v1",
}

_DEFAULT_API_KEY_ENVS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "bailian": "DASHSCOPE_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "alibaba": "DASHSCOPE_API_KEY",
    "xiaomi": "XIAOMI_API_KEY",
}


def _non_thinking_request_options(model_name: str) -> dict[str, object]:
    """Request fields that disable optional reasoning for the model family of
    ``model_name`` (matched by name prefix below).

    Thinking-only families raise ``ValueError``; the other matched families
    get ``enable_thinking=False`` or ``thinking.type=disabled``; any other
    model gets no extra fields.
    """
    model = model_name.lower()
    thinking_only = (
        model.startswith(("deepseek-r1", "kimi-k2-thinking"))
        or (model.startswith("qwen") and "thinking" in model)
    )
    if thinking_only:
        raise ValueError(
            f"Model {model_name!r} is thinking-only and cannot be used in "
            "non-thinking experiments."
        )
    if model.startswith("qwen3") or model.startswith(
        ("deepseek-v3.1", "deepseek-v3.2", "deepseek-v4-")
    ):
        # llama.cpp servers read only chat_template_kwargs.enable_thinking (a
        # top-level enable_thinking is silently ignored); sending both fields
        # switches reasoning off on hosted and llama.cpp endpoints alike.
        return {"extra_body": {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }}
    if model.startswith("mimo-v2") and "-tts" not in model:
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    if model == "kimi-k2.5":
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def _non_thinking_temperature(model_name: str, temperature: float | None) -> float:
    """Temperature of a non-thinking request: ``temperature`` (default 0.0),
    except for a model that accepts only one fixed temperature (0.6)."""
    if model_name.lower() == "kimi-k2.5":
        return 0.6
    return 0.0 if temperature is None else temperature


def _explicit_thinking_request_options(
    *,
    model_name: str,
    platform: str,
    thinking_enabled: bool,
) -> dict[str, object]:
    """Map an explicit thinking on / off setting to provider-specific request
    fields, chosen by platform name or model-name prefix: ``thinking.type``
    (enabled / disabled) or ``enable_thinking`` (top level and in
    ``chat_template_kwargs``); other providers get no extra fields."""
    model = model_name.lower()
    provider = platform.lower()
    if provider == "deepseek" or model.startswith("deepseek-v4"):
        thinking_type = "enabled" if thinking_enabled else "disabled"
        return {"extra_body": {"thinking": {"type": thinking_type}}}
    if provider in {"bailian", "dashscope", "qwen", "alibaba"} or model.startswith(
        "qwen"
    ):
        return {"extra_body": {
            "enable_thinking": thinking_enabled,
            "chat_template_kwargs": {"enable_thinking": thinking_enabled},
        }}
    if provider == "xiaomi" or model.startswith("mimo-v2"):
        thinking_type = "enabled" if thinking_enabled else "disabled"
        return {"extra_body": {"thinking": {"type": thinking_type}}}
    return {}


class OpenAIChatClient:
    """Small OpenAI chat-completions wrapper.

    The ``openai`` package is imported only when a client is built, so code
    and tests that never build one run without it, network access or API
    keys.  Use this client for real LLM calls.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key_env: str | None = None,
        platform: str = "openai",
        thinking_enabled: bool | None = None,
        max_retries: int | None = None,
        max_completion_tokens: int | None = None,
        require_provider_usage: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        """Build the underlying ``openai.OpenAI`` client.

        ``base_url`` / ``api_key_env`` override the platform's preset (the
        ``openai`` platform has none: the SDK then reads ``OPENAI_BASE_URL``
        / ``OPENAI_API_KEY``); a resolved key variable that is unset raises
        ``RuntimeError``, as does a missing ``openai`` package.  HTTP
        timeouts: ``OPENAI_TIMEOUT`` (read / write / pool, default 120 s) and
        ``OPENAI_CONNECT_TIMEOUT`` (connect, default the smaller of 10 s and
        ``OPENAI_TIMEOUT``); retries come from ``OPENAI_MAX_RETRIES``
        (default 2) unless ``max_retries`` is given.  The HTTP client keeps
        no cookies unless ``OPENAI_KEEP_COOKIES`` is ``1`` or ``true``.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the openai package to use OpenAIChatClient"
            ) from exc
        timeout_total = float(os.environ.get("OPENAI_TIMEOUT", "120"))
        # Fail fast on an endpoint that cannot be reached or stalls during
        # connection set-up (TCP connect, TLS handshake): the short connect
        # timeout abandons it quickly instead of waiting out the longer
        # OPENAI_TIMEOUT.  The wall-clock guard (TimeoutLLMClient) still
        # bounds the whole call.
        connect_timeout = float(
            os.environ.get("OPENAI_CONNECT_TIMEOUT", str(min(10.0, timeout_total)))
        )
        http_client: object | None = None
        try:
            import httpx

            timeout: object = httpx.Timeout(timeout_total, connect=connect_timeout)

            class _NoCookieHTTPClient(httpx.Client):
                """An httpx client whose cookie jar never persists anything.

                httpx re-wraps any ``cookies=`` argument into a plain jar, so
                a Cookies subclass cannot opt out; pinning the property is
                the reliable seam.  The API needs no cookies, but a load
                balancer that sets session cookies makes the default jar
                grow across the thousands of calls a long run makes in one
                process, until the Cookie header alone exceeds the server's
                header limit (HTTP 431) and every request fails.
                OPENAI_KEEP_COOKIES=1 restores the default client.
                """

                @property
                def cookies(self):  # noqa: D102
                    return httpx.Cookies()

                @cookies.setter
                def cookies(self, value):  # noqa: D102
                    return None

            if os.environ.get("OPENAI_KEEP_COOKIES", "").strip() not in {"1", "true"}:
                http_client = _NoCookieHTTPClient(timeout=timeout)
        # Fallback: the SDK's own HTTP client, with OPENAI_TIMEOUT for every
        # phase (connect included).
        except Exception:  # pragma: no cover - httpx ships with the openai SDK
            timeout = timeout_total
        resolved_max_retries = (
            int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
            if max_retries is None
            else max_retries
        )
        if (
            isinstance(resolved_max_retries, bool)
            or not isinstance(resolved_max_retries, int)
            or resolved_max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        if max_completion_tokens is not None and (
            isinstance(max_completion_tokens, bool)
            or not isinstance(max_completion_tokens, int)
            or max_completion_tokens < 1
        ):
            raise ValueError("max_completion_tokens must be a positive integer")
        if not isinstance(require_provider_usage, bool):
            raise TypeError("require_provider_usage must be boolean")
        self._platform = platform.lower()
        self._thinking_enabled = thinking_enabled
        # OpenAI reasoning models take reasoning_effort and reject sampling
        # temperature; when an effort is pinned the request omits temperature.
        # The env fallback lets subprocess-built clients (python sandbox
        # workers, whose payload schema carries no effort field) pin an
        # effort for reasoning worker models.
        self._reasoning_effort = (
            reasoning_effort
            if reasoning_effort is not None
            else (os.environ.get("OPENAI_REASONING_EFFORT") or None)
        )
        # Global output cap from OPENAI_MAX_COMPLETION_TOKENS when the caller
        # passes none: unbounded thinking on llama.cpp-class servers can drag
        # a single call past the HTTP timeout.  Unset or invalid: no cap.
        if max_completion_tokens is None:
            _env_cap = os.environ.get("OPENAI_MAX_COMPLETION_TOKENS")
            if _env_cap:
                try:
                    _cap = int(_env_cap)
                    max_completion_tokens = _cap if _cap >= 1 else None
                except ValueError:
                    pass
        self._max_completion_tokens = max_completion_tokens
        self._require_provider_usage = require_provider_usage
        client_kwargs: dict[str, object] = {
            "timeout": timeout,
            "max_retries": resolved_max_retries,
        }
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        resolved_base_url = base_url or _DEFAULT_BASE_URLS.get(self._platform)
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        resolved_api_key_env = api_key_env or _DEFAULT_API_KEY_ENVS.get(self._platform)
        if resolved_api_key_env:
            api_key = os.environ.get(resolved_api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{resolved_api_key_env} is required for {platform} client"
                )
            client_kwargs["api_key"] = api_key
        self._client = OpenAI(**client_kwargs)

    def complete(
        self,
        prompt: str,
        model_name: str,
        temperature: float | None = None,
        json_mode: bool = True,
    ) -> LLMResponse:
        """One chat-completions call; returns the text and the token usage.

        With ``thinking_enabled`` unset, the request switches optional
        reasoning off for the model families that
        :func:`_non_thinking_request_options` knows; otherwise the explicit
        setting is mapped to provider fields.  A pinned reasoning
        effort replaces the temperature.  Usage the provider does not report
        is estimated (an error under ``require_provider_usage``).
        """
        if self._thinking_enabled is None:
            request_options = _non_thinking_request_options(model_name)
            request_temperature = _non_thinking_temperature(model_name, temperature)
        else:
            request_options = _explicit_thinking_request_options(
                model_name=model_name,
                platform=self._platform,
                thinking_enabled=self._thinking_enabled,
            )
            request_temperature = 0.0 if temperature is None else temperature

        # json_mode (default) constrains output to a single JSON object (empty
        # content then reads as "{}"); callers that need free text, such as
        # planner calls returning Python source and message_only_v2 worker
        # calls, pass json_mode=False.
        if json_mode:
            request_options = {
                **request_options,
                "response_format": {"type": "json_object"},
            }
        if self._max_completion_tokens is not None:
            request_options = {
                **request_options,
                "max_completion_tokens": self._max_completion_tokens,
            }
        if self._reasoning_effort is not None:
            request_options = {
                **request_options,
                "reasoning_effort": self._reasoning_effort,
            }
            response = self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **request_options,
            )
        else:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=request_temperature,
                **request_options,
            )
        text = response.choices[0].message.content or ("{}" if json_mode else "")
        usage = response.usage
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        prompt_usage_known = bool(
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and prompt_tokens >= 0
        )
        completion_usage_known = bool(
            isinstance(completion_tokens, int)
            and not isinstance(completion_tokens, bool)
            and completion_tokens >= 0
        )
        provider_usage_known = prompt_usage_known and completion_usage_known
        if self._require_provider_usage and not provider_usage_known:
            raise RuntimeError(
                "provider-authoritative prompt/completion usage is required"
            )
        return LLMResponse(
            text=text,
            usage=LLMUsage(
                prompt_tokens=(
                    prompt_tokens
                    if prompt_usage_known
                    else estimate_tokens(prompt)
                ),
                completion_tokens=(
                    completion_tokens
                    if completion_usage_known
                    else estimate_tokens(text)
                ),
            ),
        )
