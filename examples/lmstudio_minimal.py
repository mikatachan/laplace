from __future__ import annotations

import asyncio

from laplace import ContentionBroker, Reaper
from laplace.adapters.lmstudio import LMStudioAdapter


async def main() -> None:
    adapter = LMStudioAdapter()
    reaper = Reaper(adapter, keep_loaded={"nomic-embed-text"})
    broker = ContentionBroker(adapter, reaper)
    reaper.start_sweep_loop()

    model_id = "qwen/qwen3-coder-30b"
    context_length = 32768
    decision = await broker.admit(model_id, context_length=context_length, priority="interactive")
    if decision != "admit":
        print("backpressure")
        return

    reaper.acquire(model_id)
    try:
        await adapter.ensure_loaded(model_id, context_length)
        print(f"admitted and loaded: {model_id}")
    finally:
        reaper.release(model_id)
        await broker.done(model_id)
        reaper.stop_sweep_loop()


if __name__ == "__main__":
    asyncio.run(main())
