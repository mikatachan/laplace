# Extraction plan: axis-discord/Laplace -> standalone `laplace`

The engine runs in production inside `axis-discord` (the clawdbot Discord bot).
This repo is the engine-agnostic extraction. Source files (in axis-discord):
`services/contention_broker.py`, `services/reaper.py`, `services/priority.py`,
`services/brokered_call.py`, `utils/lmstudio_unload.py`, and the narrow-waist in
`agents/openclaw_relay.py`. Design log: `.discuss/discuss-20260618-081500-*.md`.

## The boundary to define first: `InferenceAdapter`

Today the broker/reaper depend on `LMStudioUnloader` directly. Step 1 is to lift
that into an abstract adapter the core depends on:

```
class InferenceAdapter(Protocol):
    async def list_loaded(self) -> list[LoadedModel]   # id, size_bytes, status, last_used_ms, queued, context_length
    async def footprint_mb(self, model_id) -> int | None
    async def ensure_loaded(self, model_id, context_length) -> None
    async def force_unload(self, model_id) -> bool      # VERIFIED (re-checks the model is actually gone)
    # identity helpers for runtimes with duplicate instances (LM Studio :N)
    def base_id(self, model_id) -> str
```

`LoadedModel.status` ("idle"/"generating"), `last_used_ms`, `queued` are the
external-tenant activity signal — the part that makes external reaping safe.

## Port order
1. `priority.py` — already engine-agnostic; copy nearly as-is. Drop the bot config-override coupling or make it a generic dict.
2. Define `adapter.py` (the Protocol above) + `LoadedModel` dataclass.
3. `adapters/lmstudio.py` — wrap the `lms` CLI + `lms ps --json` (lift from `utils/lmstudio_unload.py`: load/unload/list/verify + the `:N` two-tier identity + the size/status/lastUsedTime/queued parsing).
4. `reaper.py` — lift sweep/TTL/`reap_duplicates`/`request_free` + the external-idle eval (`_eval_external_idle`, `_EXTERNAL_IDLE_TTL`, sustained-idle, pre-unload re-check). Replace direct unloader calls with `self.adapter`.
5. `broker.py` — lift the heap priority queue + `admit`/`done`/`notify_freed` + lock-free eviction. Depend on the adapter (residency reads) + the reaper (request_free).
6. (optional) `admission.py` — the engine-agnostic narrow-waist contextmanager pattern, decoupled from OpenClawRelayAgent, as a helper others wrap at their chokepoint.
7. Port the tests (`test_contention_broker`, `test_reaper`, `test_priority`) against a `FakeAdapter`.
8. `examples/lmstudio_minimal.py` — a runnable demo.

## Adapters to add (the open-source seed)
- `adapters/ollama.py` — `ollama ps` / `/api/ps` (expires_at, size), `ollama stop`; note Ollama already idle-unloads via keep_alive, so its value is the priority/admission layer.
- `adapters/vllm.py` — likely thin/N/A (one model per process); document why.

## Keep in sync
axis-discord keeps using its in-tree copy for now. If this extraction proves out,
axis-discord could depend on `laplace` instead (a later migration).
