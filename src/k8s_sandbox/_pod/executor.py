from __future__ import annotations

import asyncio
import contextvars
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

from inspect_ai.util import concurrency

from k8s_sandbox._logger import log_debug, log_warn
from k8s_sandbox._pod.timing import POD_OPERATION_TIMING, PodOperationTiming

T = TypeVar("T")

_DEFAULT_SLOW_OP_SECONDS = 5.0


def _slow_op_threshold_seconds() -> float:
    """Seconds at or above which an operation is logged at WARNING, not DEBUG."""
    try:
        return float(os.environ["INSPECT_POD_OP_SLOW_SECONDS"])
    except (KeyError, ValueError):
        return _DEFAULT_SLOW_OP_SECONDS


class PodOpExecutor:
    """
    A singleton class that manages a thread pool executor for running pod operations.

    This class's API is asynchronous, but the operations it runs are synchronous. It
    runs operations in a thread pool executor.

    Interacts with Inspect's concurrency context manager for the purpose of displaying
    the number of ongoing operations.
    """

    _instance: PodOpExecutor | None = None

    def __init__(self, max_pod_ops: int | None = None) -> None:
        if max_pod_ops is not None:
            self._max_workers = max_pod_ops
            source = "max_pod_ops argument"
        else:
            try:
                self._max_workers = int(os.environ["INSPECT_MAX_POD_OPS"])
                source = "INSPECT_MAX_POD_OPS env var"
            except (KeyError, ValueError):
                cpu_count = os.cpu_count() or 1
                # Pod operations are typically I/O-bound (from the
                # client's perspective).
                self._max_workers = cpu_count * 4
                source = f"default (cpu_count={cpu_count} * 4)"
        log_debug(
            "Creating PodOpExecutor.",
            max_workers=self._max_workers,
            source=source,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers, thread_name_prefix="pod-op-executor"
        )
        # Instrumentation counters. Only ever mutated from the event loop (never
        # from a worker thread), so they need no lock.
        self._waiting = 0
        self._running = 0
        self._slow_op_seconds = _slow_op_threshold_seconds()

    @classmethod
    def get_instance(cls, max_pod_ops: int | None = None) -> PodOpExecutor:
        """Gets the singleton instance of the PodOpExecutor.

        Args:
            max_pod_ops: Maximum number of concurrent pod operations. If provided
                on the first call, overrides the INSPECT_MAX_POD_OPS env var and
                the default (cpu_count * 4). A later call with a different value
                raises ValueError rather than silently ignoring the configuration.

        This method is async-safe (because it doesn't await anything) but not
        thread-safe.
        """
        if cls._instance is None:
            cls._instance = cls(max_pod_ops=max_pod_ops)
        elif max_pod_ops is not None and cls._instance._max_workers != max_pod_ops:
            raise ValueError(
                "PodOpExecutor is already initialized with "
                f"max_pod_ops={cls._instance._max_workers}; cannot use "
                f"max_pod_ops={max_pod_ops}."
            )
        return cls._instance

    async def queue_operation(self, callable: Callable[[], T]) -> T:
        """
        Queue a synchronous pod operation to run asynchronously and return the result.

        A thread pool executor is used to run the operation in another thread.

        Inspect's concurrency context manager is used so that the user gets visibility
        of the number of ongoing operations. Other than the user display, the
        use of the semaphore is redundant.

        Records how long the operation spent in each stage so that a slow operation
        can be attributed: waiting for the semaphore, waiting for a free worker
        thread, or in the Kubernetes call itself.

        This method is async-safe but not thread-safe.
        """
        submitted_at = time.monotonic()
        self._waiting += 1
        acquired = False
        try:
            async with concurrency("pod-op", self._max_workers):
                self._waiting -= 1
                acquired = True
                acquired_at = time.monotonic()
                queued = self._waiting
                self._running += 1
                # run_in_executor does not propagate the caller's context into the
                # worker thread, so pass it directly to preserve Inspect
                # sandbox config overrides
                context = contextvars.copy_context()
                started_at: float | None = None

                def run_op() -> T:
                    nonlocal started_at
                    started_at = time.monotonic()
                    _ = context.run(POD_OPERATION_TIMING.set, None)
                    return context.run(callable)

                try:
                    return await asyncio.get_event_loop().run_in_executor(
                        self._executor, run_op
                    )
                finally:
                    finished_at = time.monotonic()
                    running = self._running
                    self._running -= 1
                    pod_operation_timing = context.get(POD_OPERATION_TIMING)
                    self._log_op_timing(
                        submitted_at=submitted_at,
                        acquired_at=acquired_at,
                        started_at=started_at,
                        finished_at=finished_at,
                        queued=queued,
                        running=running,
                        pod_operation_timing=pod_operation_timing,
                    )
        finally:
            if not acquired:
                self._waiting -= 1

    def _log_op_timing(
        self,
        *,
        submitted_at: float,
        acquired_at: float,
        started_at: float | None,
        finished_at: float,
        queued: int,
        running: int,
        pod_operation_timing: PodOperationTiming | None,
    ) -> None:
        total = finished_at - submitted_at
        fields: dict[str, object] = {
            "total_s": round(total, 3),
            # Time blocked on the pod-op semaphore: high means the concurrency
            # limit is the constraint.
            "semaphore_wait_s": round(acquired_at - submitted_at, 3),
            # Time between submitting to the thread pool and the callable actually
            # starting: high means every worker thread is busy.
            "dispatch_wait_s": (
                round(started_at - acquired_at, 3) if started_at is not None else None
            ),
            # Time in the synchronous Kubernetes call itself.
            "call_s": (
                round(finished_at - started_at, 3) if started_at is not None else None
            ),
            "connect_s": (
                round(pod_operation_timing.connect_s, 3)
                if pod_operation_timing is not None
                else None
            ),
            "command_s": (
                round(pod_operation_timing.command_s, 3)
                if pod_operation_timing is not None
                else None
            ),
            "queued": queued,
            "running": running,
            "max_workers": self._max_workers,
        }
        if total >= self._slow_op_seconds:
            log_warn("Slow pod operation.", **fields)
        else:
            log_debug("Pod operation timing.", **fields)
