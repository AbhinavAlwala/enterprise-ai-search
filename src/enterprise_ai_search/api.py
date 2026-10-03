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
        started = perf_counter()
        try:
            service = service_factory()
        except (OSError, ValueError, RuntimeError) as error:
            logger.error("Service initialization failed (%s)", type(error).__name__)
            service = SearchService(None, None, None, perf_counter() - started)
        app.state.service = service
        try:
            yield
        finally:
            service.close()
            app.state.service = None
            logger.info("Service resources released")

    app = FastAPI(title="Enterprise AI Search", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        # Default validation errors include submitted values, which may contain secrets.
        details = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.middleware("http")
    async def unexpected_error(request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as error:
            # A final HTTP boundary avoids leaking exception details in server tracebacks.
            logger.error("Unexpected request error (%s)", type(error).__name__)
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    def ready_service(request: Request) -> SearchService:
        service = request.app.state.service
        if service is None or not service.retrieval_initialized or not service.authorization_initialized:
            raise HTTPException(status_code=503, detail="Service unavailable")
        return service

    def authorized_service(request: Request) -> tuple[SearchService, PrincipalContext]:
        names = ("x-tenant-id", "x-principal-id", "x-groups")
        values = [request.headers.getlist(name) for name in names]
        if len(values[0]) != 1 or len(values[1]) != 1 or len(values[2]) > 1:
            raise HTTPException(status_code=400, detail="Invalid request identity")
        try:
            principal = parse_principal(values[0][0], values[1][0], values[2][0] if values[2] else None)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid request identity") from None
        service = ready_service(request)
        if principal.tenant_id not in service.policy_store.tenants:
            raise HTTPException(status_code=403, detail="Access denied")
        return service, principal

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
            logger.error("Search request failed (%s)", type(error).__name__)
            raise HTTPException(status_code=500, detail="Search request failed") from None
        hits = [
            SearchHit(rank=r.rank, document_id=r.document_id, chunk_id=r.passage.chunk_id,
                      text=r.passage.text, reranker_score=r.score)
            for r in results
        ]
        handler_seconds = perf_counter() - started
        logger.info("Search completed in %.3f s; results=%d", handler_seconds, len(hits))
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
            logger.error("Answer request failed (%s)", type(error).__name__)
            raise HTTPException(status_code=502, detail="Answer request failed") from None
        timings = {**result.timings, "handler_seconds": perf_counter() - started}
        logger.info("Answer completed in %.3f s; citation validation=%s",
                    timings["handler_seconds"], result.citation_validation.passed)
        return AskResponse(answer=result.answer, evidence=list(result.evidence),
                           citation_validation=result.citation_validation, timings=timings)

    return app
