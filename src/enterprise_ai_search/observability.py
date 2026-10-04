import json
import logging
import math
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from uuid import RFC_4122, UUID, uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from enterprise_ai_search.generation import Generator

logger = logging.getLogger(__name__)
ROUTES = {("GET", "/health"): "health", ("GET", "/metrics"): "metrics",
          ("POST", "/search"): "search", ("POST", "/ask"): "ask"}
ROUTE_NAMES = (*ROUTES.values(), "other")
STATUS_FAMILIES = ("1xx", "2xx", "3xx", "4xx", "5xx", "other")
STAGES = ("retrieval_reranking_seconds", "generation_request_seconds", "online_seconds")
METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"}


def request_id(headers: list[tuple[bytes, bytes]]) -> str:
    values = [value for name, value in headers if name.lower() == b"x-request-id"]
    if len(values) == 1 and len(values[0]) == 36:
        try:
            value = values[0].decode("ascii")
            parsed = UUID(value)
            if str(parsed) == value and parsed.version == 4 and parsed.variant == RFC_4122:
                return value
        except (UnicodeDecodeError, ValueError):
            pass
    return str(uuid4())


def log_event(log: logging.Logger, event: str, *, level: int = logging.INFO, **fields: object) -> None:
    data = {"event": event, **fields}
    log.log(level, json.dumps(data, allow_nan=False), extra=data)


@dataclass
class RequestTrace:
    request_id: str
    method: str
    route: str
    authorization_denied: bool = False
    unexpected_error: bool = False
    generation_failed: bool = False
    timings: dict[str, float] = field(default_factory=dict)


current_trace: ContextVar[RequestTrace | None] = ContextVar("request_trace", default=None)


def record_timings(timings: dict[str, float]) -> None:
    trace = current_trace.get()
    if trace is not None:
        for name in STAGES:
            value = timings.get(name)
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                trace.timings[name] = value


def mark_authorization_denied() -> None:
    trace = current_trace.get()
    if trace is not None:
        trace.authorization_denied = True


def log_request_error(log: logging.Logger, error: Exception, *, unexpected: bool) -> None:
    trace = current_trace.get()
    if trace is not None:
        trace.unexpected_error |= unexpected
    log_event(log, "request_error", level=logging.ERROR,
              request_id=trace.request_id if trace else None, error_type=type(error).__name__,
              unexpected=unexpected)


@dataclass
class Duration:
    count: int = 0
    total: float = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self.total += seconds

    def snapshot(self) -> dict[str, int | float]:
        return {"count": self.count, "total": self.total,
                "mean": self.total / self.count if self.count else 0.0}


class Metrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._duration = Duration()
        self._route_durations = {route: Duration() for route in ROUTE_NAMES}
        self._statuses = {route: dict.fromkeys(STATUS_FAMILIES, 0) for route in ROUTE_NAMES}
        self._stages = {name: Duration() for name in STAGES}
        self._denied = self._unexpected = self._generation_failed = 0

    def record(self, trace: RequestTrace, status: int, seconds: float) -> None:
        route = trace.route if trace.route in ROUTE_NAMES else "other"
        family = f"{status // 100}xx" if 100 <= status < 600 else "other"
        with self._lock:
            self._duration.add(seconds)
            self._route_durations[route].add(seconds)
            self._statuses[route][family] += 1
            self._denied += int(trace.authorization_denied)
            self._unexpected += int(trace.unexpected_error)
            self._generation_failed += int(trace.generation_failed)
            for name in STAGES:
                if name in trace.timings:
                    self._stages[name].add(trace.timings[name])

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "total_requests": self._duration.count,
                "search_requests": self._route_durations["search"].count,
                "ask_requests": self._route_durations["ask"].count,
                "authorization_denied_requests": self._denied,
                "unexpected_errors": self._unexpected,
                "generation_failures": self._generation_failed,
                "request_duration_seconds": self._duration.snapshot(),
                "requests_by_route": {
                    route: {"status_families": dict(self._statuses[route]),
                            "duration_seconds": self._route_durations[route].snapshot()}
                    for route in ROUTE_NAMES
                },
                "stages_seconds": {name: duration.snapshot() for name, duration in self._stages.items()},
            }


class ObservedGenerator:
    def __init__(self, generator: Generator) -> None:
        self.generator = generator

    def generate(self, messages: list[dict[str, str]]) -> str:
        trace = current_trace.get()
        if trace is None:
            return self.generator.generate(messages)
        started = perf_counter()
        try:
            answer = self.generator.generate(messages)
            if not isinstance(answer, str) or not answer.strip():
                trace.generation_failed = True
            return answer
        except Exception:
            trace.generation_failed = True
            raise
        finally:
            record_timings({"generation_request_seconds": perf_counter() - started})


class RequestTracingMiddleware:
    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = perf_counter()
        method = scope["method"] if scope["method"] in METHODS else "OTHER"
        trace = RequestTrace(request_id(scope.get("headers", [])), method,
                             ROUTES.get((method, scope["path"]), "other"))
        token = current_trace.set(trace)
        status = 0

        async def traced_send(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = [(name, value) for name, value in message.get("headers", [])
                           if name.lower() != b"x-request-id"]
                message = {**message, "headers": [*headers, (b"x-request-id", trace.request_id.encode("ascii"))]}
            await send(message)

        try:
            await self.app(scope, receive, traced_send)
        except Exception as error:
            log_request_error(logger, error, unexpected=True)
            if status:
                # Once headers are sent, a second response would violate the HTTP protocol.
                raise RuntimeError("Response failed after headers sent") from None
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
            await response(scope, receive, traced_send)
        finally:
            seconds = perf_counter() - started
            self.metrics.record(trace, status, seconds)
            log_event(logger, "request_completed", request_id=trace.request_id, method=trace.method,
                      route=trace.route, status=status, duration_seconds=seconds, timings=trace.timings,
                      authorization_denied=trace.authorization_denied,
                      unexpected_error=trace.unexpected_error, generation_failed=trace.generation_failed)
            current_trace.reset(token)
