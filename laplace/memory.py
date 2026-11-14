"""Live unified-memory headroom for admission decisions (macOS)."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Awaitable, Callable, List, Optional

log = logging.getLogger(__name__)

_MIB = 1024 * 1024
_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_ROW_RE = re.compile(r'^"?([^":]+)"?:\s+(\d+)\.?\s*$')

Runner = Callable[[List[str]], Awaitable[Optional[str]]]


async def _run(args: list[str]) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("laplace.memory: %s failed: %s", args[0], exc)
        return None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
    except Exception as exc:  # noqa: BLE001 - includes timeout
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        log.warning("laplace.memory: %s failed: %s", args[0], exc)
        return None
    if proc.returncode != 0:
        log.warning("laplace.memory: %s exited rc=%s", args[0], proc.returncode)
        return None
    return (out or b"").decode()


def parse_vm_stat(text: str) -> dict[str, int]:
    """Return vm_stat page counts keyed by row label, plus ``page_size``."""
    match = _PAGE_SIZE_RE.search(text)
    if not match:
        raise ValueError("vm_stat output has no page size")
    rows: dict[str, int] = {"page_size": int(match.group(1))}
    for line in text.splitlines():
        row = _ROW_RE.match(line.strip())
        if row:
            rows[row.group(1).strip()] = int(row.group(2))
    return rows


class VmStatHeadroom:
    """Callable returning live headroom in MB, or ``None`` when unreadable.

    Headroom is free + speculative pages + reclaimable file-backed pages, less
    a safety margin on the file-cache credit and compressor growth since the
    previous reading. Speculative pages are removed from the file-backed pool
    because they are already counted directly. When ``iogpu.wired_limit_mb``
    is set, headroom is also capped by the room left under that limit, since
    model memory is GPU-wired.
    """

    def __init__(
        self, run: Runner | None = None, file_cache_reclaim_margin: float = 0.05
    ):
        self._run = run or _run
        self._file_cache_reclaim_margin = file_cache_reclaim_margin
        self._last_compressor_pages: int | None = None

    async def __call__(self) -> int | None:
        text = await self._run(["vm_stat"])
        if text is None:
            return None
        try:
            rows = parse_vm_stat(text)
            page = rows["page_size"]
            free = rows["Pages free"]
            speculative = rows.get("Pages speculative", 0)
            active = rows["Pages active"]
            inactive = rows["Pages inactive"]
            file_backed = rows.get("File-backed pages", 0)
            wired = rows["Pages wired down"]
            compressor = rows.get("Pages occupied by compressor", 0)
        except (KeyError, ValueError) as exc:
            log.warning("laplace.memory: unparseable vm_stat output: %s", exc)
            return None

        growth = 0
        if self._last_compressor_pages is not None:
            growth = max(0, compressor - self._last_compressor_pages)
        self._last_compressor_pages = compressor

        reclaimable_file = min(active + inactive, max(0, file_backed - speculative))
        file_credit = int(
            reclaimable_file * (1.0 - self._file_cache_reclaim_margin)
        )
        file_margin = reclaimable_file - file_credit
        reclaimable = free + speculative + file_credit - growth
        headroom_mb = reclaimable * page // _MIB

        limit_text = await self._run(["sysctl", "-n", "iogpu.wired_limit_mb"])
        limit = (limit_text or "").strip()
        wired_cap_mb: int | None = None
        if limit.isdigit() and int(limit) > 0:
            wired_cap_mb = int(limit) - wired * page // _MIB
            headroom_mb = min(headroom_mb, wired_cap_mb)
        result_mb = max(0, headroom_mb)
        log.debug(
            "laplace.memory: free_pages=%s speculative_pages=%s "
            "file_credit_pages=%s file_margin_pages=%s "
            "file_cache_reclaim_margin=%s "
            "compressor_growth_pages=%s wired_cap_mb=%s result_mb=%s",
            free,
            speculative,
            file_credit,
            file_margin,
            self._file_cache_reclaim_margin,
            growth,
            wired_cap_mb,
            result_mb,
        )
        return result_mb
