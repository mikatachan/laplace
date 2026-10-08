"""Entry point: python -m laplace_serve --config PATH."""

from __future__ import annotations

import argparse
import logging
import os
from logging.handlers import RotatingFileHandler

from aiohttp import web

from laplace.adapters.lmstudio import LMStudioAdapter
from laplace.broker import ContentionBroker
from laplace.reaper import Reaper

from laplace_serve.app import build_app
from laplace_serve.config import LaplacedConfig

log = logging.getLogger(__name__)


def build_components(
    config: LaplacedConfig,
) -> tuple[LMStudioAdapter, Reaper, ContentionBroker]:
    """Wire adapter -> reaper -> broker from config.

    Footprint overrides and load timeout flow into the adapter; keep-loaded set
    into the reaper; budget, concurrency cap, and admit timeout into the broker.
    """
    adapter = LMStudioAdapter(
        load_timeout_s=config.load_timeout_s,
        load_grace_s=config.load_grace_s,
        footprint_overrides=dict(config.model_footprint_mb) or None,
        parallel_overrides=dict(config.model_parallel) or None,
    )
    reaper = Reaper(adapter, keep_loaded=set(config.keep_loaded))
    broker = ContentionBroker(
        adapter,
        reaper,
        budget_mb=config.budget_mb_or_none(),
        timeout_s=config.admit_timeout_s,
        max_concurrent_lms=config.max_concurrent,
    )
    return adapter, reaper, broker


def _configure_logging(config: LaplacedConfig) -> None:
    path = config.resolved_log_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=config.log_rotate_mb * 1024 * 1024,
        backupCount=config.log_backups,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="laplace_serve")
    parser.add_argument(
        "--config",
        default=os.path.expanduser("~/.laplace/laplaced.toml"),
        help="path to laplaced.toml",
    )
    args = parser.parse_args(argv)

    config = LaplacedConfig.from_toml_file(args.config)
    _configure_logging(config)

    adapter, reaper, broker = build_components(config)
    app = build_app(config, adapter=adapter, reaper=reaper, broker=broker)

    log.info(
        "laplaced starting on %s:%s -> upstream %s", config.bind, config.port, config.upstream
    )
    web.run_app(app, host=config.bind, port=config.port, print=None)


if __name__ == "__main__":
    main()
