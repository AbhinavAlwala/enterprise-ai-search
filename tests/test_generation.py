import io
import json
from urllib.error import HTTPError, URLError

import pytest

from enterprise_ai_search import generation
from enterprise_ai_search.generation import GenerationConfig, HttpGenerator


def test_environment_config_requires_endpoint_and_model_without_echoing_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GENERATION_ENDPOINT", "GENERATION_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GENERATION_API_KEY", "fixture-secret")
    with pytest.raises(ValueError, match="GENERATION_ENDPOINT, GENERATION_MODEL") as error:
        GenerationConfig.from_env()
    assert "fixture-secret" not in str(error.value)
    monkeypatch.setenv("GENERATION_ENDPOINT", "http://localhost:8000/v1/chat/completions")
    monkeypatch.setenv("GENERATION_MODEL", " fixture-model ")
    config = GenerationConfig.from_env()
    assert config.model == "fixture-model" and config.api_key == "fixture-secret"
    assert "fixture-secret" not in repr(config)


@pytest.mark.parametrize("endpoint", ["", "file:///tmp/model", "localhost:8000", "https://", "http://user:password@localhost/v1", "http://localhost:invalid/v1", "http://localhost/v1#fragment"])
def test_invalid_endpoint_rejected(endpoint: str) -> None:
    with pytest.raises(ValueError, match="GENERATION_ENDPOINT"):
        GenerationConfig(endpoint, "model")


@pytest.mark.parametrize("change", [{"model": " "}, {"timeout_seconds": 0}, {"timeout_seconds": float("nan")}, {"max_output_tokens": 0}])
def test_invalid_configuration_rejected(change: dict) -> None:
    values = {"endpoint": "http://localhost/v1/chat/completions", "model": "model", **change}
    with pytest.raises(ValueError):
        GenerationConfig(**values)


@pytest.mark.parametrize("api_key", ["fixture-secret\nmore", "fixture-secret\rmore", "fixture-secret€"])
def test_invalid_auth_header_configuration_never_echoes_key(api_key: str) -> None:
    with pytest.raises(ValueError, match="GENERATION_API_KEY") as error:
        GenerationConfig("http://localhost/v1", "model", api_key)
    assert "fixture-secret" not in str(error.value)


@pytest.mark.parametrize("api_key", ["", "fixture-secret"])
def test_nonstreaming_http_request_and_response_without_network(monkeypatch: pytest.MonkeyPatch, api_key: str) -> None:
    messages = [{"role": "system", "content": "Use evidence."}, {"role": "user", "content": "Question with café"}]
    def fake_open(request: object, timeout: float) -> io.BytesIO:
        assert request.full_url == "http://localhost:8000/v1/chat/completions"
        assert request.get_method() == "POST"
        assert request.get_header("Content-type") == "application/json"
        assert request.get_header("Authorization") == ("Bearer " + api_key if api_key else None)
        assert timeout == 60
        assert json.loads(request.data) == {"model": "fixture-model", "messages": messages, "stream": False, "max_tokens": 512}
        return io.BytesIO(json.dumps({"choices": [{"message": {"content": "Fixture café [1]"}}]}).encode())
    monkeypatch.setattr(generation, "urlopen", fake_open)
    client = HttpGenerator(GenerationConfig("http://localhost:8000/v1/chat/completions", "fixture-model", api_key))
    assert client.generate(messages) == "Fixture café [1]"


@pytest.mark.parametrize("response", [b"not JSON", b"\xff", b"{}", b'{"choices": []}', b'{"choices": [{"message": {"content": null}}]}', b'{"choices": [{"message": {"content": " "}}]}', b"[]"])
def test_malformed_or_empty_response_rejected(monkeypatch: pytest.MonkeyPatch, response: bytes) -> None:
    monkeypatch.setattr(generation, "urlopen", lambda *args, **kwargs: io.BytesIO(response))
    with pytest.raises(ValueError):
        HttpGenerator(GenerationConfig("http://localhost/v1", "model")).generate([])


@pytest.mark.parametrize("error", [
    HTTPError("http://localhost/fixture-secret", 401, "fixture-secret", {}, io.BytesIO(b"fixture-secret")),
    URLError("fixture-secret"), TimeoutError("fixture-secret"),
])
def test_request_errors_do_not_echo_server_body_credentials_or_url(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error
    monkeypatch.setattr(generation, "urlopen", fail)
    with pytest.raises(ValueError, match="Generation") as failure:
        HttpGenerator(GenerationConfig("http://localhost/v1", "model", "fixture-secret")).generate([])
    assert "fixture-secret" not in str(failure.value)
