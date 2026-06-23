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

## Prior art / where Laplace fits

See the assessment in `docs/`. Closest adjacent tools: **llama-swap** (own-process idle TTL), **ConductorAPI** (multi-runtime budgets+priority, early-stage), **ollamaMQ** (Ollama request-priority proxy), **LocalAI** (LRU + idle watchdog). None do external-tenant-aware reaping.

## Limitations
- Leans on the runtime's activity-signal schema (`lms ps --json` fields), which is undocumented and could drift; fails safe (unknown means don't reap).
- Shared-model gap: a model used by both you and an external process can't be perfectly distinguished by identity alone.
- Not yet a published package; adapter extraction in progress (`EXTRACTION.md`).

## License
MIT.
