"""Opt-in caller authentication for the governed REST surface."""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Mapping

from aiohttp import web

log = logging.getLogger(__name__)


class CallerAuthenticator:
    """Map high-entropy bearer tokens to configured origins.

    Verifiers use ``sha256:<salt-hex>:<digest-hex>`` where the digest is
    SHA-256(salt || token). Tokens are random credentials, not passwords; the
    salt prevents verifier reuse and ``compare_digest`` avoids early exits.
    """

    def __init__(self, verifiers: Mapping[str, str]):
        self._verifiers = tuple(
            (origin, *_parse_verifier(verifier))
            for origin, verifier in sorted(verifiers.items())
        )

    def authenticate(
        self, request: web.Request, path_origin: str | None
    ) -> web.Response | None:
        token = _bearer_token(request.headers.get("Authorization"))
        matched_origin = self._match(token) if token is not None else None
        route_class = _route_class(request.path)
        if matched_origin is None:
            log.info(
                "caller_auth origin=- route=%s decision=unauthorized", route_class
            )
            return web.json_response({"error": "caller_auth_required"}, status=401)
        if path_origin is not None and matched_origin != path_origin:
            log.info(
                "caller_auth origin=%s route=%s decision=origin_mismatch",
                matched_origin,
                route_class,
            )
            return web.json_response({"error": "caller_origin_mismatch"}, status=403)
        log.info(
            "caller_auth origin=%s route=%s decision=allow",
            matched_origin,
            route_class,
        )
        return None

    def _match(self, token: str) -> str | None:
        token_bytes = token.encode()
        matched_origin = None
        for origin, salt, expected in self._verifiers:
            actual = hashlib.sha256(salt + token_bytes).digest()
            if hmac.compare_digest(actual, expected):
                matched_origin = origin
        return matched_origin


def make_verifier(token: str, salt: bytes) -> str:
    """Return verifier material for tests and an operator-side helper."""
    digest = hashlib.sha256(salt + token.encode()).hexdigest()
    return f"sha256:{salt.hex()}:{digest}"


def validate_verifier(value: str) -> None:
    _parse_verifier(value)


def _parse_verifier(value: str) -> tuple[bytes, bytes]:
    try:
        algorithm, salt_hex, digest_hex = value.split(":", 2)
        salt = bytes.fromhex(salt_hex)
        digest = bytes.fromhex(digest_hex)
    except (AttributeError, ValueError):
        raise ValueError("invalid caller verifier") from None
    if algorithm != "sha256" or len(salt) < 16 or len(digest) != 32:
        raise ValueError("invalid caller verifier")
    return salt, digest


def _bearer_token(value: str | None) -> str | None:
    if not value:
        return None
    scheme, separator, token = value.partition(" ")
    if (
        separator
        and scheme.lower() == "bearer"
        and token
        and all(0x21 <= ord(char) <= 0x7E for char in token)
    ):
        return token
    return None


def _route_class(path: str) -> str:
    if path.startswith("/ops/"):
        return "operations"
    if path.endswith("/v1/models"):
        return "models"
    if path in {"/readyz", "/fleet/status"}:
        return "status"
    return "inference"
