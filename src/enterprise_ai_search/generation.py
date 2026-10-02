import json
import math
import os
from dataclasses import dataclass, field
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class GenerationConfig:
    endpoint: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 60.0
    max_output_tokens: int = 512
    temperature: float | None = None

    def __post_init__(self) -> None:
        try:
            url = urlsplit(self.endpoint)
            valid_url = url.scheme in ("http", "https") and bool(url.hostname) and not url.username and not url.password and not url.fragment
            url.port
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError("GENERATION_ENDPOINT must be an HTTP(S) URL without embedded credentials or a fragment")
        if not self.model.strip():
            raise ValueError("GENERATION_MODEL must be nonempty")
        if any(not 32 <= ord(character) < 127 for character in self.api_key):
            raise ValueError("GENERATION_API_KEY must contain printable ASCII characters")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0 or self.max_output_tokens <= 0:
            raise ValueError("Generation timeout and output token limit must be positive")
        if self.temperature is not None and (not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
            raise ValueError("Generation temperature must be finite and between 0 and 2")

    @classmethod
    def from_env(cls) -> "GenerationConfig":
        endpoint = os.environ.get("GENERATION_ENDPOINT", "").strip()
        model = os.environ.get("GENERATION_MODEL", "").strip()
        missing = [name for name, value in (("GENERATION_ENDPOINT", endpoint), ("GENERATION_MODEL", model)) if not value]
        if missing:
            raise ValueError("Missing generation configuration: " + ", ".join(missing))
        return cls(endpoint, model, os.environ.get("GENERATION_API_KEY", "").strip())


class HttpGenerator:
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.config.model, "messages": messages, "stream": False,
            "max_tokens": self.config.max_output_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        request = Request(self.config.endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            # Endpoint error bodies can echo credentials or the full evidence prompt.
            raise ValueError(f"Generation endpoint returned HTTP {error.code}") from None
        except (URLError, OSError):
            raise ValueError("Generation request failed: connection error or timeout") from None
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("Generation endpoint did not return valid UTF-8 JSON") from None
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError("Generation response must include choices[0].message.content") from None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Generation response must contain nonempty answer text")
        return content
