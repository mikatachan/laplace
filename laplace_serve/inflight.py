"""Opt-in, in-memory progress records for in-flight proxied chat completions.

A watcher can read what a long streamed call is producing without the proxy
buffering or altering the stream: the relay writes each chunk to the client
first, then hands the same bytes here. Records hold counters, timestamps and a
fixed-size tail of the streamed text deltas. Tail text is never logged.
"""

from __future__ import annotations

import itertools
import json
import time

from aiohttp import web

# An SSE line longer than this is never JSON-parsed, and a partial line longer
# than this is dropped (with the rest of that line skipped), so one line cannot
# grow the record's memory or parse cost.
MAX_LINE_BYTES = 64 * 1024


class InflightRecord:
    """Progress of one proxied request. Mutated only by the relay loop."""

    def __init__(self, origin: str, model: str | None, stream: bool, tail_chars: int):
        self.origin = origin
        self.model = model
        self.stream = stream
        self.started_at = time.time()
        self.admitted_at: float | None = None
        self.first_byte_at: float | None = None
        self.last_chunk_at: float | None = None
        self.bytes = 0
        self.chunks = 0
        self.tail = ""
        self.observer_failed = False
        self._tail_chars = tail_chars
        self._pending = b""
        self._discarding = False

    def mark_admitted(self) -> None:
        self.admitted_at = time.time()

    def observe(self, chunk: bytes) -> None:
        """Account for a chunk already written to the client."""
        if not chunk:
            return
        now = time.time()
        if self.first_byte_at is None:
            self.first_byte_at = now
        self.last_chunk_at = now
        self.bytes += len(chunk)
        self.chunks += 1
        if self.stream:
            self._feed(chunk)

    def _feed(self, chunk: bytes) -> None:
        if self._discarding:
            newline = chunk.find(b"\n")
            if newline < 0:
                return
            self._discarding = False
            chunk = chunk[newline + 1:]
        *lines, rest = (self._pending + chunk).split(b"\n")
        for line in lines:
            self._parse_line(line)
        if len(rest) > MAX_LINE_BYTES:
            self._pending = b""
            self._discarding = True
        else:
            self._pending = rest

    def _parse_line(self, line: bytes) -> None:
        if len(line) > MAX_LINE_BYTES:
            return
        line = line.rstrip(b"\r")
        if not line.startswith(b"data:"):
            return
        data = line[len(b"data:"):].strip()
        if not data or data == b"[DONE]":
            return
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError, RecursionError):
            # RecursionError: deeply nested JSON from a malformed stream.
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list):
            return
        for choice in payload["choices"]:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                self._append(content)
            reasoning = delta.get("reasoning_content")
            if not isinstance(reasoning, str):
                reasoning = delta.get("reasoning")
            if isinstance(reasoning, str):
                self._append(reasoning)

    def _append(self, text: str) -> None:
        if text:
            self.tail = (self.tail + text[-self._tail_chars:])[-self._tail_chars:]

    def snapshot(self) -> dict:
        return {
            "origin": self.origin,
            "model": self.model,
            "stream": self.stream,
            "started_at": self.started_at,
            "admitted_at": self.admitted_at,
            "first_byte_at": self.first_byte_at,
            "last_chunk_at": self.last_chunk_at,
            "bytes": self.bytes,
            "chunks": self.chunks,
            "tail": self.tail,
            "observer_failed": self.observer_failed,
        }


# The relay finds the record for its request here, so neither the local proxy
# nor remote backends need a new parameter.
RECORD = web.RequestKey("inflight_record", InflightRecord)


class InflightRegistry:
    """In-flight records keyed by an opaque id; empty when the feature is off."""

    def __init__(self, origins: frozenset[str] = frozenset(), tail_chars: int = 400):
        self.origins = frozenset(origins)
        self.tail_chars = tail_chars
        self._records: dict[int, InflightRecord] = {}
        self._ids = itertools.count()

    def start(self, origin: str, model: str | None, stream: bool) -> tuple[int, InflightRecord] | None:
        if origin not in self.origins:
            return None
        key = next(self._ids)
        record = InflightRecord(origin, model, stream, self.tail_chars)
        self._records[key] = record
        return key, record

    def finish(self, key: int) -> None:
        self._records.pop(key, None)

    def snapshot(self) -> dict:
        return {
            "inflight": [record.snapshot() for record in self._records.values()],
            "now": time.time(),
        }
