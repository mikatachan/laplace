# Laplace

**Priority-aware admission control plus idle and external-tenant model reaping for a *shared* local LLM inference pool.**

If you run multiple LLM workloads (agents, a bot, n8n/cron jobs) against **one** local inference server (LM Studio, Ollama) on a machine with **finite memory** (Mac unified memory, a single GPU), you hit memory contention: models pile up and OOM, idle models squat VRAM, concurrent loads collide, and a background job blocks an interactive one because there's no priority. Laplace manages that pool as a finite, contended resource.

> Status: **early / extracting.** The design runs in production (a multi-agent Discord bot on an M4 Ultra plus LM Studio). This repo is the standalone, engine-adapter extraction of that core. The LM Studio adapter is the reference; Ollama and vLLM adapters are the next to add.

---

## What's new

A [prior-art scan](#prior-art--where-laplace-fits) found that the *basic* pieces (an admission cap and idle TTL for models **you** loaded) are commodity (LM Studio's own TTL/Auto-Evict, Ollama `keep_alive`, llama-swap, LocalAI, llama.cpp router). What no surveyed tool does:

1. **Reap *external* models**, models loaded by **other processes** sharing the same server, safely, using the server's **own per-model activity signal** (e.g. LM Studio `lms ps --json` `status`/`lastUsedTime`/`queued`), with a sustained-idle gate and a pre-unload re-check. Every existing tool only manages models *it* loaded or processes *it* spawned. Laplace is a background observer/reaper that co-exists with other independent tenants on the pool.
2. **3-tier admission priority** (interactive / scheduled / background), *server-derived*, so a background load is held until interactive capacity is confirmed, not just a flat queue.

If you don't have other processes sharing your server and you just want idle-unload, your runtime already does that, so you may not need this.

---

## Core (engine-agnostic) plus adapters

```
your dispatch ── admit(model, ctx, priority) ─► Broker (memory-fit + heap priority queue + backpressure)
                                                   │ request_free ▲ notify_freed
                                                   ▼              │
                                                 Reaper (sweep, tiered TTL, external-tenant reap)
                                                   │ via
                                                   ▼
                                            InferenceAdapter  ◄── LMStudioAdapter (reference)
                                            (list_loaded / footprint /            OllamaAdapter (todo)
                                             ensure_loaded / force_unload /        vLLMAdapter (n/a)
                                             activity: status/last_used/queued)
```

- **Broker**: `admit()` gates on memory fit; heap priority queue; 503-style backpressure; evicts only via the Reaper.
- **Reaper**: 60s sweep, tiered idle TTL, and the external-tenant reap (the distinctive part). Sole owner of unloads.
- **Priority**: 3 server-derived tiers.
- **InferenceAdapter**: the small engine boundary you implement per runtime.

## Engine fit by runtime

| | LM Studio | Ollama | vLLM |
|---|---|---|---|
| Shared multi-model pool | yes | yes | no (one model per `serve`) |
| Built-in idle unload | no, Laplace owns it | **yes** (`keep_alive`) | n/a |
| **What Laplace adds** | the whole stack | **priority admission + external-tenant reap** (Ollama lacks priority) | doesn't fit the pool model |

Best fit: **LM Studio-style shared multi-model servers with multiple independent clients.** Useful on Ollama for the *priority/admission* layer. Not a fit for single-model-per-process vLLM.

## Install

```sh
pip install .          # regular install
pip install -e .       # editable (dev); needs pip >= 21.3 (PEP 660); run `pip install -U pip` if older
```

Stdlib-only core (no runtime deps); `pytest`/`pytest-asyncio` only for the test suite. Verified end-to-end against live LM Studio: a fresh venv with nothing but this package installed can admit, load, and reap a real model.

## Quick start (LM Studio)

```python
from laplace import ContentionBroker, Reaper
from laplace.adapters.lmstudio import LMStudioAdapter

adapter = LMStudioAdapter()
reaper = Reaper(adapter, keep_loaded={"nomic-embed-text"})
broker = ContentionBroker(adapter, reaper, budget_mb=87040)
reaper.start_sweep_loop()

decision = await broker.admit(model_id, context_length, priority="interactive")
if decision == "backpressure":
    ...  # 503 / retry

reaper.acquire(model_id)
try:
    await adapter.ensure_loaded(model_id, context_length)
    ...
finally:
    reaper.release(model_id)
    await broker.done(model_id)
```

*(API is settling as the extraction lands; see `EXTRACTION.md`.)*

### Optional remote backends

`laplaced.toml` may declare an OpenAI-compatible remote backend. A backend is
inert until a `[placement]` entry names it; `axis` is the built-in local
backend. Credentials are obtained from `auth_command` at startup and after a
401, retained only in process memory, and never written to status or logs.

```toml
[backends.shinobi]
url = "http://shinobi.example:1234"
alt_urls = ["http://shinobi-tailnet.example:1234"]
slots = 1
probe_model = "google/gemma-4-e4b"
auth_command = ["credential-helper", "read", "shinobi"]

[placement]
"bot:google/gemma-4-e4b" = ["shinobi", "axis"]
```

Remote state and fixed reason codes are available at `GET /fleet/status`.
Omitting `[placement]` performs no credential command, probe, or remote request.

### Opt-in governed REST authentication

Caller authentication and local-upstream credential injection are disabled by
default. Merely installing or restarting this revision therefore preserves the
existing routes and Authorization forwarding. Enforcement starts only when the
top-level boolean below is explicitly present and true:

```toml
governed_rest_auth = true
upstream_auth_command = ["credential-helper", "read", "axis.laplaced.upstream"]

[origins]
hermes = "interactive"
bot = "interactive"
default = "scheduled"

[caller_verifiers]
hermes = "sha256:<at-least-16-byte-salt-hex>:<sha256-digest-hex>"
bot = "sha256:<at-least-16-byte-salt-hex>:<sha256-digest-hex>"
```

The operator-side provisioning helper must generate each verifier from a
separate random bearer token without printing that token. Configuration must
contain exactly one verifier for every non-default origin; raw caller tokens
and the LM Studio key never belong in TOML. Enabling the flag without complete
verifiers or a fixed-argv upstream helper fails startup.

When enabled, only origin-prefixed inference and models routes exist. The
bearer token must map to the path origin; missing/invalid credentials return a
fixed 401 and cross-origin credentials return a fixed 403. `/readyz` and
`/fleet/status` also require any configured caller token, while `/healthz`
remains unauthenticated liveness. Laplaced strips the caller Authorization
header and injects the helper-provided upstream credential. Bare `/v1/...`
routes are absent.

When the same flag is enabled, three authenticated operational probes are
also registered:

- `GET /ops/upstream-listener` performs only a bounded TCP connect to the local
  LM Studio REST port and reports whether it is listening.
- `GET /ops/upstream-api` requires an authenticated `200` from `/v1/models`
  with an OpenAI-compatible list shape.
- `GET /ops/catalog/{model}` runs a bounded `lms ls --json` inventory and
  distinguishes an available model, a missing model, and a control-plane
  failure.

The aggregate `/readyz` uses the strict API probe only in enabled mode. With
the flag absent or false, these routes do not exist and the legacy readiness
semantics—including treating any upstream HTTP response as reachability—remain
unchanged.

### Opt-in in-flight progress

`GET /inflight` lists in-progress proxied `/v1/chat/completions` requests from
configured origins, so a watcher can tell a long stream that is producing text
from one that is stuck. It is off by default: with `origins` empty (or the
table absent) no record is kept and the endpoint returns an empty list.

```toml
[inflight]
origins = ["hermes"]
tail_chars = 400
```

```json
{"inflight": [{"origin": "hermes", "model": "m", "stream": true,
  "started_at": 1762540000.1, "admitted_at": 1762540000.2,
  "first_byte_at": 1762540001.0, "last_chunk_at": 1762540042.5,
  "bytes": 18234, "chunks": 311, "tail": "...last 400 characters",
  "observer_failed": false}],
 "now": 1762540043.0}
```

Times are Unix epoch seconds; `admitted_at` is null for remote backends.
`tail` holds the last `tail_chars` characters of the streamed
`delta.content` and `delta.reasoning_content` (or `delta.reasoning`) text,
parsed from complete SSE `data:` lines; unparseable lines still count toward
`bytes`; lines over 64 KiB are not parsed. Each chunk is recorded only after it
is written to the client, so the relayed stream is unchanged. If recording a
chunk ever raises, the stream continues untouched, `observer_failed` becomes
true and that record stops updating. A record is removed when its request ends, for any
reason. Records live in memory only, and tail text is never logged. The
endpoint answers loopback peers only (403 otherwise) and, with
`governed_rest_auth`, needs the same caller token as `/fleet/status`.

### Opt-in live-memory admission

By default a cold load is admitted when the static `[model_footprint_mb]`
figures sum under the budget. That sum ignores context, parallel slots and
what else is using memory. With the flag below, laplaced instead admits a
cold load only when its cost fits in live headroom:

```toml
live_memory_admission = true
```

- **Cost** = weights + KV + overhead. Weights come from the `lms ls --json`
  `sizeBytes` for the model, or from `[model_footprint_mb]` when the catalog
  has no figure. KV = per-layer K+V bytes from the model's `config.json` (under
  `~/.lmstudio/models/<path>`) x context length (sliding layers stop at their
  window; linear-attention layers are skipped) x parallel (`[model_parallel]`).
  Overhead is 10% of weights.
- **Headroom** is read from `vm_stat` at each decision: free + speculative +
  the inactive pages that are surely file-backed, minus compressor growth since
  the previous reading, capped by the room left under `iogpu.wired_limit_mb`.
  `vm_stat` reports inactive and file-backed as separate totals, so only their
  guaranteed overlap is counted: `inactive + file-backed - (active + inactive +
  speculative)`, floored at 0. Anonymous inactive pages need compression to
  reclaim and are never counted.
- **Reserved first**: the full KV a busy resident model can still grow to at
  its loaded context (busy = in flight through laplaced, or `lms ps` reports it
  as not idle, with unknown status, or with queued requests), plus the full
  cost of admitted loads that are not resident yet.
- **Busy models are never evicted** on this path, whatever laplaced's own
  refcount says.
- **Context must be covered**: a request rides on a resident instance or a
  pending load only when that instance's context is at least the request's.
  A larger context than a pending load waits; a larger context than a resident
  instance is a new cost decision.
- **Architecture lookup**: `config.json` is found under the models directory via
  the catalog entry's `indexedModelIdentifier`, then its `path`. Hub entries whose
  path is not on disk fall back to the conservative KV estimate (256 KiB per
  token per sequence, i.e. 32768 MiB at 131072 context), which can hold loads in
  the queue; the log names it `kv=no-architecture`.

If the cost does not fit, idle models are evicted when their weights would
cover the shortfall. Freed memory is measured with `vm_stat`, not inferred
from file size. Otherwise the request waits for `admit_timeout_s` and then
gets the usual 503. An unreadable `vm_stat` or an unknown cost also means
wait, never load. Each decision logs one `load decision` line with every term;
`basis=estimate(...)` names any term that was not derived from observed data
(footprint-table weights, missing `config.json`, unpinned parallel, sized as 4).
Requests for a model already resident at a sufficient context do not need a
new load and are admitted as before.

### Default-off dynamic parallel sizing

Dynamic sizing is an opt-in extension of live-memory admission:

```toml
live_memory_admission = true
dynamic_load_sizing = true

[model_parallel]
"qwen/qwen3-coder-30b" = 2
```

At each cold load, the configured `[model_parallel]` value is a ceiling rather
than a forced allocation. laplaced counts the current request plus queued
waiters for the same model, then chooses the largest parallel value no greater
than that demand whose explicit load cost fits authoritative live headroom
after busy-KV and pending-load reserves. Configured context is unchanged. If
parallel 1 does not fit, the request follows the existing eviction and
backpressure path. When a model has no `[model_parallel]` entry, dynamic sizing
uses the adapter's existing fallback parallel (4 for LM Studio) as the ceiling,
matching the default used by the non-dynamic live-admission path.

Load-decision logs include demand, selected parallel, authoritative headroom,
and a clearly labelled `shadow_headroom_mb`. The shadow value expands the
file-backed reclaim estimate with protected, dirty/unavailable, and margin
terms set to zero for observation only. It never authorizes a load.

To roll back, set `dynamic_load_sizing = false` and restart laplaced. Existing
resident models retain the dimensions at which they were loaded until they are
unloaded normally.

## Prior art / where Laplace fits

See the assessment in `docs/`. Closest adjacent tools: **llama-swap** (own-process idle TTL), **ConductorAPI** (multi-runtime budgets+priority, early-stage), **ollamaMQ** (Ollama request-priority proxy), **LocalAI** (LRU + idle watchdog). None do external-tenant-aware reaping.

## Limitations
- Leans on the runtime's activity-signal schema (`lms ps --json` fields), which is undocumented and could drift; fails safe (unknown means don't reap).
- Shared-model gap: a model used by both you and an external process can't be perfectly distinguished by identity alone.
- Not yet a published package; adapter extraction in progress (`EXTRACTION.md`).
- Client disconnect closes the upstream request: for a stream, when the next chunk write fails; before upstream response headers (all of a non-stream generation), within `DISCONNECT_POLL_S` (1 s, `laplace_serve/proxy.py`). LM Studio stops generating when that connection closes, but its log says it finishes prompt processing first (observed for streams).

## License
MIT.

LM Studio load mutations are serialized within each adapter. A fresh read-only
residency check runs before the lock and any failure guard: a single matching
instance with sufficient context admits even while another model is loading.
Cold loads require a second successful snapshot under the lock and an exact
`lms ls --json` catalog key. Failed inventory reads never mean empty inventory.
Existing instances (including `:N` identifiers and matching model keys/paths)
prevent another load. Insufficient or missing context fails admission; it is
rechecked on each request, not cached. Automatic context upgrades are not done:
this adapter cannot prove an instance is idle for external clients, and has no
atomic reservation-and-unload interface. An operator must reconcile that instance.

While the reaper unloads a model, `lms ps` can still list it. The broker queues
new requests for that model (`reason=unload-in-progress`) instead of riding on the
leaving instance, and an admitted request waits for the unload, including its
retry, to return before it loads or proxies. Each reaper unload re-checks that no
admission has pinned the victim since it was selected.

A timed-out CLI is killed and reaped. Timeouts, cancellation, nonzero exits, and
successful commands without a visible result retain the load's monotonic start
time. Until `load_timeout_s + load_grace_s` has elapsed from that start, unresolved
loads block cold loads of **any** model: pending server work is absent from the
broker's resident-memory budget. Resident admissions still use the fast path.
`load_grace_s` defaults to 90 seconds and is configurable on `LMStudioAdapter` and
as a top-level `laplaced.toml` option.

Each admission takes a fresh inventory. A matching resident reconciles uncertainty;
after the deadline, no new matching or unidentified instance marks the attempt
failed and permits retry. Failed inventory reads retain the restriction. Known
other catalog models do not prevent recovery. A flaky verification read recovers
when the requested model is present, including before the deadline.

A possible fuzzy result disables retries for that requested ID until sufficient
residency is verified. The failure record then clears; unrelated catalog changes
do not permit retries. No snapshot difference proves ownership, so verification never unloads
new instances automatically, including unknown identifiers. Known other catalog
keys are excluded from fuzzy candidates. Operator cleanup may therefore be needed.

Bounded retry assumes server work becomes visible or stops within the configured
window. Killing `lms` does not establish cancellation of server-side work, and an
empty `ps` cannot prove that no server load is pending. The grace window is an
operational bound, not a server cancellation guarantee; choose it accordingly.
An unidentified new instance remains blocked for operator reconciliation even
after the deadline. This is a process-local guard; concurrent external loads of
the SAME model require coordination beyond this adapter.

Startup logs ERRORs for configured context/footprint keys absent from the catalog,
including the closest key, but continues serving.

### Model ids in requests: send the full LM Studio id

`[model_context]`, `[model_footprint_mb]`, `[model_parallel]` and `keep_loaded`
are keyed by the LM Studio model key as `lms ls` reports it. Callers must send
that full id in the request. laplaced matches an exact key first, then a bare id
that equals the segment after the last `/` of exactly one configured key (for
example `qwen3-coder-30b` for `qwen/qwen3-coder-30b`). It does not match
prefixes or aliases. An id that matches nothing logs `unconfigured model id` and
is used unchanged, with `default_context_length` (64000 by default) instead of
its configured context.

Example: with `"text-embedding-nomic-embed-text-v1.5" = 2048` in
`[model_context]`, a request for `text-embedding-nomic-embed-text-v1.5` loads the
embedder at 2048. A request for the shortened `nomic-embed-text` matches no key
and loads it at 64000.
