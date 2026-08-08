import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

from inspect_ai._util.logger import TRACE  # TODO: Using private package.
from inspect_ai.util import trace_action, trace_message

logger = logging.getLogger(__name__)

TRUNCATED_SUFFIX = "...<truncated-for-logging>"
# The threshold at which to truncate individual arguments in logging messages.
# Some ExecResults can contain very large outputs.
DEFAULT_ARG_TRUNCATION_THRESHOLD = 1000

# Formatting is skipped entirely when the destination level is disabled. It is not a
# micro-optimisation: every sandbox operation formats its kwargs, and on a busy eval-set
# runner that dominates the process. A py-spy sampling profile of a production runner
# that had stopped answering its Inspect control channel put ~26% of all samples in
# _format_kwargs_as_json, with the abc/inspect machinery it drives accounting for
# most of the rest -- against ~18% for the actual TLS and WebSocket work. It burned
# ~1 core of GIL with 31 cores idle and 325 of 326 threads asleep, so the asyncio event
# loop could not get enough GIL time to service its own socket.
#
# All of that work was discarded: the runner logs at WARNING (effective level 30, no
# root handlers), so TRACE and DEBUG were both disabled and every formatted string went
# straight to a no-op logger call. Measured at 11.8us per call on a live runner.
#
# These checks are per-call, so raising the log level at runtime still produces full
# detail. log_error/log_warn are deliberately not gated: they always emit.


def log_trace(message: str, **kwargs: Any) -> None:
    """Format and log a message at TRACE level with K8s category.

    Args:
        message: The log message.
        **kwargs: Key-value pairs to include in the log message. Values are truncated if
          they exceed DEFAULT_ARG_TRUNCATION_THRESHOLD (which can be overridden with env
          var INSPECT_K8S_LOG_TRUNCATION_THRESHOLD).
    """
    if not logger.isEnabledFor(TRACE):
        return
    formatted = format_log_message(message, **kwargs)
    trace_message(logger, category="K8s", message=formatted)


def log_debug(message: str, **kwargs: Any) -> None:
    """Format and log a message at DEBUG level with K8s prefix.

    Args:
        message: The log message.
        **kwargs: Key-value pairs to include in the log message. Values are truncated if
          they exceed DEFAULT_ARG_TRUNCATION_THRESHOLD (which can be overridden with env
          var INSPECT_K8S_LOG_TRUNCATION_THRESHOLD).
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    formatted = format_log_message(message, **kwargs)
    logger.debug(f"K8s: {formatted}")


def log_error(message: str, **kwargs: Any) -> None:
    """Format and log a message at ERROR level with K8s prefix.

    Args:
        message: The log message.
        **kwargs: Key-value pairs to include in the log message. Values are truncated if
          they exceed DEFAULT_ARG_TRUNCATION_THRESHOLD (which can be overridden with env
          var INSPECT_K8S_LOG_TRUNCATION_THRESHOLD).
    """
    formatted = format_log_message(message, **kwargs)
    logger.error(f"K8s: {formatted}")


def log_warn(message: str, **kwargs: Any) -> None:
    """Format and log a message at WARNING level with K8s prefix.

    Args:
        message: The log message.
        **kwargs: Key-value pairs to include in the log message. Values are truncated if
          they exceed DEFAULT_ARG_TRUNCATION_THRESHOLD (which can be overridden with env
          var INSPECT_K8S_LOG_TRUNCATION_THRESHOLD).
    """
    formatted = format_log_message(message, **kwargs)
    logger.warning(f"K8s: {formatted}")


def format_log_message(message: str, **kwargs: Any) -> str:
    """Format message in a structured fashion.

    Args:
        message: The log message.
        **kwargs: Key-value pairs to include in the log message. Values are truncated if
          they exceed DEFAULT_ARG_TRUNCATION_THRESHOLD (which can be overridden with env
          var INSPECT_K8S_LOG_TRUNCATION_THRESHOLD).
    """
    if not kwargs:
        return message
    json_kwargs = _format_kwargs_as_json(**kwargs)
    return f"{message} {json_kwargs}"


@contextmanager
def inspect_trace_action(action: str, **kwargs: Any) -> Generator[None, None, None]:
    """Context manager that traces an action with structured logging.

    Uses Inspect's trace_action.

    Args:
        action: The action being performed (e.g. "K8s execute command in Pod").
        **kwargs: Key-value pairs to include in as details parameter to trace_action.
          Values are truncated if they exceed DEFAULT_ARG_TRUNCATION_THRESHOLD (which
          can be overridden with env var INSPECT_K8S_LOG_TRUNCATION_THRESHOLD).
    """
    # This is the hot path: it wraps EVERY pod operation via _log_op.
    #
    # trace_action is purely observational -- every branch does nothing but
    # logger.log(TRACE, ...) and it ends in a bare `raise`, so control flow is
    # identical either way. With TRACE off it therefore produces no output at all,
    # and the whole thing can be skipped: the kwargs formatting, the uuid, the
    # monotonic timers, and traceback.format_exc() on the error path.
    #
    # Gating only the formatting removes 48% of the per-op cost; skipping the
    # context manager as well removes 95% (16.0us -> 0.7us, measured).
    if not logger.isEnabledFor(TRACE):
        yield
        return

    json_kwargs = _format_kwargs_as_json(**kwargs)
    with trace_action(logger, action, json_kwargs):
        yield


def _truncate_arg(arg: Any) -> str:
    arg_str = str(arg)
    truncation_threshold = _get_arg_truncation_threshold()
    if len(arg_str) > truncation_threshold:
        return arg_str[:truncation_threshold] + TRUNCATED_SUFFIX
    return arg_str


def _get_arg_truncation_threshold() -> int:
    try:
        return int(os.environ["INSPECT_K8S_LOG_TRUNCATION_THRESHOLD"])
    except (KeyError, ValueError):
        return DEFAULT_ARG_TRUNCATION_THRESHOLD


def _format_kwargs_as_json(**kwargs: Any) -> str:
    truncated_kwargs = {k: _truncate_arg(v) for k, v in kwargs.items()}
    return json.dumps(truncated_kwargs, ensure_ascii=False)
