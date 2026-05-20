"""Structured logging via structlog.

Usage::

    from vlm_anomaly.logging import get_logger

    log = get_logger(__name__)
    log.info("backend.predict", model="mock", latency_ms=1.2)

Renderer selection:
    - ``LOG_FORMAT=json``  → JSON lines (production / Kaggle notebooks).
    - anything else        → human-readable console output (development).

Call :func:`configure_logging` once at process startup (the CLI and
scripts do this automatically).  Library code should only call
:func:`get_logger` and never call ``configure_logging`` itself.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(json_logs: bool = False, log_level: str = "INFO") -> None:
    """Wire up structlog with the appropriate renderer.

    Uses structlog's stdlib integration so ``structlog.get_logger()`` and
    ``logging.getLogger()`` share one handler chain.

    Args:
        json_logs: Emit JSON lines when ``True``; colourful console when ``False``.
        log_level: Standard ``logging`` level name (e.g. ``"DEBUG"``).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(log_level.upper())


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for ``name``.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A structlog :class:`~structlog.stdlib.BoundLogger` instance.
    """
    return structlog.get_logger(name)
