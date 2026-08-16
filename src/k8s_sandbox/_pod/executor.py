from __future__ import annotations

import asyncio
import contextvars
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, TypeVar

import anyio
from inspect_ai.util import concurrency

from k8s_sandbox._logger import log_debug, log_warn
from k8s_sandbox._pod.timing import POD_OPERATION_TIMING, PodOperationTiming

T = TypeVar("T")

# Seconds at or above which a pod operation is logged at WARNING, not DEBUG.
_SLOW_OP_SECONDS = 5.0

# How long a cancelled operation's worker is given to finish once its transport
# has been closed. Waking it is near-instant when the socket is live; this is
# the ceiling for a transport that does not wake at all.
SETTLE_AFTER_CANCEL_SECONDS = 30.0


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
        self._slow_op_seconds = _SLOW_OP_SECONDS

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

    async def queue_operation(
        self, callable: Callable[[], T], on_cancel: Callable[[], None] | None = None
    ) -> T:
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

                worker = self._executor.submit(run_op)
                try:
                    return await asyncio.wrap_future(worker)
                except asyncio.CancelledError:
                    # Cancelling the await does not stop the worker: it is a thread
                    # blocked on a socket read that may never return. Left alone it
                    # holds a pool slot the next operation needs, keeps the process
                    # from exiting (ThreadPoolExecutor joins its threads at
                    # interpreter exit), and may still write into a destination the
                    # caller is about to dispose. So wake it by closing its
                    # transport, then wait for it to finish before unwinding.
                    # Without a transport to close there is nothing to wake the
                    # worker with: waiting would tax the cancelling caller for the
                    # whole settle bound and change nothing, and such operations
                    # (e.g. the pod-restart check) write to no caller-owned
                    # destination, so there is nothing to settle FOR.
                    if on_cancel is not None:
                        on_cancel()
                        await self._settle(worker)
                    raise
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

    @staticmethod
    async def _settle(worker: Future[T]) -> None:
        """Wait for a cancelled operation's worker to finish, within a bound.

        The caller is unwinding from cancellation, and under an anyio cancel
        scope (how inspect_ai cancels, e.g. `anyio.move_on_after` around a
        transcript drain) the backend re-delivers that cancellation at every
        await until the scope exits -- an unshielded wait here is cancelled
        immediately and the worker is abandoned after all. Shield the wait;
        the settle bound is what keeps it finite.
        """
        with anyio.move_on_after(SETTLE_AFTER_CANCEL_SECONDS, shield=True) as scope:
            try:
                await asyncio.wrap_future(worker)
            except Exception as e:
                # The worker finished, which is all settling needs. Closing its
                # transport usually makes it finish by raising (BrokenPipeError
                # wrapped in PodError, K8sError, ...); the caller is unwinding
                # with CancelledError and will never see that exception, so
                # record what was dropped rather than swallowing it silently.
                log_debug(
                    "Cancelled pod operation's worker finished by raising; "
                    "dropping its exception in favour of the cancellation.",
                    error=repr(e),
                )
        if scope.cancelled_caught or not worker.done():
            # The transport did not wake it. Nothing further is available from
            # here, and blocking the event loop would spread one stuck socket
            # to the whole eval; say so where an operator can see it.
            log_warn(
                "Pod operation did not settle after its transport was closed; "
                "its worker thread is still running.",
                settle_timeout=SETTLE_AFTER_CANCEL_SECONDS,
            )

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
