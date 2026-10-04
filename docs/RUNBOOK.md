# Local setup and verification

Run commands from the repository root. Use Python 3.12 and uv 0.11.26 to align with CI/Docker; Python patch releases are not pinned outside the image. The package supports Windows and Linux CPU wheels. Initial dependency/corpus/encoder/reranker preparation needs network access. Tests need none of the runtime assets or a generator.

## Install and test

```sh
uv sync --locked
uv run --locked --offline pytest -q
uv build --offline
git diff --check
```

`uv.lock` is authoritative; no broad upgrade is needed. Offline sync/build additionally require packages/build tools already in uv's cache. On this Windows workspace, append `--cache-dir .uv-cache` to uv commands. Tests mock models/HTTP; an autouse fixture blocks HTTP connection creation without breaking Windows event-loop socket pairs. No formatter/linter is configured.

## Prepare runtime assets once

Unset `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and `HF_DATASETS_OFFLINE` before initial preparation, if previously set:

```sh
uv run --locked enterprise-search download
uv run --locked enterprise-search prepare-dense
uv run --locked enterprise-search prepare-reranker
```

Defaults place checksum-verified corpus/query/qrels files in `data/scifact`, the pinned encoder in `data/dense/models`, dense vectors in `data/dense/chunks.npz`, and pinned cross-encoder in `data/reranker/models`. Generated/local data is ignored. The embedding cache must match pinned model/content/configuration/runtime identity; stale caches regenerate. `uv --offline` controls package access, not model downloads, so set model offline flags after preparation.

Cached CLI query (POSIX shell):

```sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run --locked --offline enterprise-search search-reranked "Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume." --top-k 3
```

CLI queries and scientific evaluations are privileged, unscoped local commands. Do not expose them to tenants as an authorization bypass. Existing result paths are frozen; any explicitly requested future benchmark must use a new output path. Nothing here launches the 188-claim run.

## Run the API

Required policy path is explicit; Python does not load `.env`. The committed overlay is **SYNTHETIC DEMO AUTHORIZATION METADATA** for five SciFact IDs, with all unlisted documents denied. Missing/malformed policy configuration makes protected routes and health unavailable. Policy changes require restart.

POSIX shell:

```sh
export AUTHORIZATION_POLICY_PATH=config/demo_access.json
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run --locked --offline uvicorn enterprise_ai_search.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

PowerShell equivalent:

```powershell
$env:AUTHORIZATION_POLICY_PATH = "config/demo_access.json"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
uv run --locked --offline --cache-dir .uv-cache uvicorn enterprise_ai_search.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Use another terminal for requests. Identity headers assume trusted upstream authentication; direct clients can spoof them. The following values demonstrate policies, not a secure login.

```sh
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/metrics
curl -i http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-ID: tenant-a' -H 'X-Principal-ID: alice' -H 'X-Groups: researchers' \
  -H 'X-Request-ID: 5996a386-cbd8-40bd-a8b8-2e86559fe286' \
  --data '{"query":"Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.","top_k":3}'
```

PowerShell request/response ID and metrics:

```powershell
$identity = @{ "X-Tenant-ID" = "tenant-a"; "X-Principal-ID" = "alice"; "X-Groups" = "researchers"; "X-Request-ID" = "5996a386-cbd8-40bd-a8b8-2e86559fe286" }
$response = Invoke-WebRequest http://127.0.0.1:8000/search -Method Post -Headers $identity -ContentType "application/json" -Body '{"query":"Radioiodine treatment of non-toxic multinodular goitre reduces thyroid volume.","top_k":3}'
$response.Headers["X-Request-ID"]
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

For the same query with `top_k=5`, the executed demo returned A's IDs `9745001, 26026009, 37912677`; changing identity to tenant-b/bob/researchers returned `6751418, 43122426`. Do not log identity or document contents. Health reports retrieval/policy readiness and generation configuration only, without probing a model. `/metrics` belongs inside the trusted boundary; counters are per-process, reset on restart, and exclude the current poll until it completes.

Stop the local verification server with Ctrl+C. If port 8000 is already occupied, select an unused port (M10/M11 used 8001).

## Optional generation configuration

Use an already available OpenAI-compatible endpoint/model. No LLM download is required for tests or CI. Set `GENERATION_ENDPOINT` to the full non-streaming chat-completions URL, `GENERATION_MODEL` to its served model name, and optional `GENERATION_API_KEY` in the server process environment. Example host endpoint: `http://localhost:11434/v1/chat/completions`, model `qwen2.5:3b`. In PowerShell use `$env:NAME = "value"`; POSIX shells use `export NAME=value`. Never paste keys into tracked files or print resolved secrets.

`POST /ask` accepts `{"question":"..."}` with the same identity headers. It uses unchanged free-form M6 generation and inline citations, not M7's verification schema. Missing generation configuration returns 503 without disabling search. Actual generation is slower and unnecessary for checking counters/CI. Valid citations establish supplied references only; an uncited insufficiency response remains preserved and visibly flagged.

## Docker

Use Docker Desktop in Linux-container mode or a compatible Linux amd64 engine. Image build needs Python dependencies/base-image access, not prepared corpus/models/Ollama:

```sh
docker build --platform linux/amd64 --tag enterprise-ai-search:m12 .
```

Runtime does require the prepared host assets and demo policy file. Compose retains its existing image tag, one non-root worker, loopback-only host port, read-only corpus/weights/policy mounts, and writable dense cache parent. Rebuild to include final source/metadata.

```sh
docker compose config --quiet
docker compose up --build -d
docker compose ps
docker compose logs search
docker compose down --timeout 90
```

Compose explicitly forwards generation variables from the shell or its ignored `.env` interpolation file; `.env.example` is reference documentation. Unset endpoint/model default to host Ollama examples; explicitly blank values disable generation. `host.docker.internal` reaches the host on Docker Desktop; container `localhost` does not. Native Linux may need a separately reachable endpoint/host mapping. Compose sets its read-only mounted `AUTHORIZATION_POLICY_PATH` explicitly.

On Unix, ensure the non-root process can read models/corpus and write dense cache; use `CONTAINER_UID`/`CONTAINER_GID` for the host owner if necessary. Do not automatically chown host files or run as root to bypass permissions. Missing mount sources fail instead of being created. Healthchecks perform no inference, and a healthy service does not prove generator connectivity. Shutdown preserves bind-mounted files.

M9 runtime verification was reported by the owner; M10/M11 API smokes ran locally. M12 adds build-only CI verification, not a new container runtime or LLM smoke. CI execution on GitHub remains pending its first push/PR. Python dependency locks, pinned model/data revisions, version-tagged Docker bases, and immutable image digests are distinct guarantees; this repository does not promise bit-identical images. No LICENSE has been selected.
