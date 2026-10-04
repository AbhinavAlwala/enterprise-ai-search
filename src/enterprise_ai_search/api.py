import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from enterprise_ai_search.authorization import PrincipalContext, parse_principal
from enterprise_ai_search.observability import (
    Metrics, RequestTracingMiddleware, log_event, log_request_error, mark_authorization_denied, record_timings,
)
from enterprise_ai_search.rag import CitationValidation, EvidenceItem
from enterprise_ai_search.service import SearchService, load_service

logger = logging.getLogger(__name__)
MAX_TOP_K = 10
MAX_TEXT_LENGTH = 2000
QueryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_LENGTH)
]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: QueryText
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: QueryText


class HealthResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    retrieval_initialized: bool
    generation_configured: bool
    authorization_initialized: bool
    preparation_seconds: float


class SearchHit(BaseModel):
    rank: int
    document_id: str
    chunk_id: str
    text: str
    reranker_score: float


class SearchResponse(BaseModel):
    results: list[SearchHit]
    timings: dict[str, float]


class AskResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]
    citation_validation: CitationValidation
    timings: dict[str, float]


def create_app(service_factory: Callable[[], SearchService] = load_service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        # Raw access logs include untrusted URL paths/query strings; bounded request logs replace them.
        logging.getLogger("uvicorn.access").disabled = True
        log_event(logger, "initialization_started")
        started = perf_counter()
        try:
            service = service_factory()
        except (OSError, ValueError, RuntimeError) as error:
            log_event(logger, "initialization_failed", level=logging.ERROR, error_type=type(error).__name__)
            service = SearchService(None, None, None, perf_counter() - started)
        app.state.service = service
        ready = service.retrieval_initialized and service.authorization_initialized
        log_event(logger, "initialization_succeeded" if ready else "initialization_unavailable",
                  preparation_seconds=service.preparation_seconds,
                  retrieval_initialized=service.retrieval_initialized,
                  authorization_initialized=service.authorization_initialized,
                  generation_configured=service.generation_configured)
        try:
            yield
        finally:
            service.close()
            app.state.service = None
            log_event(logger, "shutdown_completed")

    app = FastAPI(title="Enterprise AI Search", version="0.1.0", lifespan=lifespan)
    app.state.metrics = Metrics()
    app.add_middleware(RequestTracingMiddleware, metrics=app.state.metrics)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        # Default validation errors include submitted values, which may contain secrets.
        details = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    def ready_service(request: Request) -> SearchService:
        service = request.app.state.service
        if service is None or not service.retrieval_initialized or not service.authorization_initialized:
            raise HTTPException(status_code=503, detail="Service unavailable")
        return service

    def authorized_service(request: Request) -> tuple[SearchService, PrincipalContext]:
        names = ("x-tenant-id", "x-principal-id", "x-groups")
        values = [request.headers.getlist(name) for name in names]
        if len(values[0]) != 1 or len(values[1]) != 1 or len(values[2]) > 1:
            mark_authorization_denied()
            raise HTTPException(status_code=400, detail="Invalid request identity")
        try:
            principal = parse_principal(values[0][0], values[1][0], values[2][0] if values[2] else None)
        except ValueError:
            mark_authorization_denied()
            raise HTTPException(status_code=400, detail="Invalid request identity") from None
        service = ready_service(request)
        if principal.tenant_id not in service.policy_store.tenants:
            mark_authorization_denied()
            raise HTTPException(status_code=403, detail="Access denied")
        return service, principal

    @app.get("/metrics")
    async def metrics() -> dict[str, object]:
        return app.state.metrics.snapshot()

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request, response: Response) -> HealthResponse:
        service = request.app.state.service
        ready = service.retrieval_initialized and service.authorization_initialized
        response.status_code = 200 if ready else 503
        return HealthResponse(
            status="ready" if ready else "not_ready", retrieval_initialized=service.retrieval_initialized,
            generation_configured=service.generation_configured,
            authorization_initialized=service.authorization_initialized,
            preparation_seconds=service.preparation_seconds,
        )

    @app.post("/search", response_model=SearchResponse)
    def search(body: SearchRequest, request: Request) -> SearchResponse:
        started = perf_counter()
        service, principal = authorized_service(request)
        try:
            results, retrieval_seconds = service.search(body.query, body.top_k, principal)
        except (OSError, ValueError, RuntimeError) as error:
            log_request_error(logger, error, unexpected=True)
            raise HTTPException(status_code=500, detail="Search request failed") from None
        hits = [
            SearchHit(rank=r.rank, document_id=r.document_id, chunk_id=r.passage.chunk_id,
                      text=r.passage.text, reranker_score=r.score)
            for r in results
        ]
        handler_seconds = perf_counter() - started
        record_timings({"retrieval_reranking_seconds": retrieval_seconds})
        return SearchResponse(
            results=hits, timings={"retrieval_reranking_seconds": retrieval_seconds,
                                  "handler_seconds": handler_seconds},
        )

    @app.post("/ask", response_model=AskResponse)
    def answer(body: AskRequest, request: Request) -> AskResponse:
        started = perf_counter()
        service, principal = authorized_service(request)
        if not service.generation_configured:
            raise HTTPException(status_code=503, detail="Generation configuration unavailable")
        try:
            result = service.answer(body.question, principal)
        except (OSError, ValueError, RuntimeError) as error:
            log_request_error(logger, error, unexpected=False)
            raise HTTPException(status_code=502, detail="Answer request failed") from None
        timings = {**result.timings, "handler_seconds": perf_counter() - started}
        record_timings(result.timings)
        return AskResponse(answer=result.answer, evidence=list(result.evidence),
                           citation_validation=result.citation_validation, timings=timings)

    return app
