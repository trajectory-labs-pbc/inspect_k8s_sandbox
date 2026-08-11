import json
import logging
import threading
import time
from abc import ABC
from dataclasses import dataclass
from typing import Generator, Literal

from kubernetes.stream import stream  # type: ignore
from kubernetes.stream.ws_client import RESIZE_CHANNEL, WSClient  # type: ignore

from k8s_sandbox._kubernetes_api import k8s_client
from k8s_sandbox._pod.error import ContainerRestartedError, PodReplacedError
from k8s_sandbox._pod.snapshot import read_pod
from k8s_sandbox._pod.timing import POD_OPERATION_TIMING, PodOperationTiming

# The duration to wait for an initial response from the k8s API server.
# The initial response is received before the command is necessarily complete, so
# long-running commands will not be affected by this timeout.
# https://github.com/kubernetes-client/python/blob/master/examples/watch/timeout-settings.md
API_TIMEOUT = 60

# Interval between WebSocket keepalive frames. Containerd's CRI streaming server
# enforces a stream_idle_timeout (default 4h [1]) that closes connections with no
# data activity. Sending a resize-channel data frame resets the idle timer via
# resetTimeout() in the server's wsstream conn.go read loop [2]. 30 seconds is well
# under any realistic idle timeout while adding negligible overhead.
#
# [1] https://github.com/kubernetes/kubernetes/blob/db9fcfeed29b860d8dd7188bc1903c4709977890/staging/src/k8s.io/kubelet/pkg/cri/streaming/server.go#L100-L105
# [2] https://github.com/kubernetes/kubernetes/blob/77b02b7ad40d36cd803856de5ba5922c947cb0aa/staging/src/k8s.io/apimachinery/pkg/util/httpstream/wsstream/conn.go#L348-L356
_KEEPALIVE_INTERVAL_SECONDS = 30
# Maximum size of a single WebSocket stdin frame. Larger single writes (tens of
# MiB) make the kubelet/API-server/TLS layer reset the connection
# (ConnectionResetError / ssl.SSLEOFError), so stdin is written in chunks.
_STDIN_CHUNK_SIZE = 1024**2  # 1 MiB

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PodInfo:
    """
    Information required to interact with a Kubernetes pod.

    This class is immutable and thread-safe.
    """

    name: str
    namespace: str
    context_name: str | None
    """The name of the kubeconfig context. If None, use the current context."""
    default_container_name: str
    uid: str
    initial_restart_count: int
    restarted_container_behavior: Literal["warn", "raise"]


class PodOperation(ABC):
    """
    A base class for a synchronous operation on a pod.

    The purpose of splitting these operations into separate classes is to encapsulate
    and isolate their respective behaviour.
    """

    _failed_to_discard_duplicate_channel = False

    def __init__(self, pod: PodInfo):
        self._pod = pod

    def _write_stdin_chunked(self, ws_client: WSClient, data: str | bytes) -> None:
        """Write ``data`` to the stdin channel in ``_STDIN_CHUNK_SIZE`` frames.

        Used by both exec and write_file (see ``_STDIN_CHUNK_SIZE`` for why we
        chunk). The slice type is preserved: ``str`` -> text frames, ``bytes``
        -> binary frames.
        """
        for i in range(0, len(data), _STDIN_CHUNK_SIZE):
            ws_client.write_stdin(data[i : i + _STDIN_CHUNK_SIZE])

    def create_websocket_client_for_exec(
        self, **kwargs
    ) -> Generator[WSClient, None, None]:
        client = k8s_client(self._pod.context_name)
        # Note: ApiException is intentionally not caught; it should fail the eval.
        stream_started_at = time.monotonic()
        ws_client: WSClient = stream(
            client.connect_get_namespaced_pod_exec,
            name=self._pod.name,
            namespace=self._pod.namespace,
            container=self._pod.default_container_name,
            _preload_content=False,
            # This is the timeout for the API request, not the command itself.
            _request_timeout=API_TIMEOUT,
            **kwargs,
        )
        stream_connected_at = time.monotonic()
        try:
            keepalive = _KEEPALIVE.register(ws_client)
            try:
                self._discard_duplicate_channel(ws_client)
                yield ws_client
            finally:
                # Unregister BEFORE close: it waits out any in-flight keepalive
                # frame, so the shared thread is never writing to a socket the
                # caller is closing (WSClient is not thread-safe).
                keepalive.unregister()
                ws_client.close()
        finally:
            _ = POD_OPERATION_TIMING.set(
                PodOperationTiming(
                    connect_s=stream_connected_at - stream_started_at,
                    command_s=time.monotonic() - stream_connected_at,
                )
            )

    def _discard_duplicate_channel(self, ws_client: WSClient) -> None:
        # Avoid issuing a warning multiple times.
        if PodOperation._failed_to_discard_duplicate_channel:
            return
        # WSClient stores all stdout and stderr in WSClient._all in addition to the
        # relevant channels. Set the _all channel to IgnoredIO to reduce memory usage.
        # https://github.com/kubernetes-client/python/issues/2302
        # Handle ImportError as we're importing a private class.
        try:
            from kubernetes.stream.ws_client import _IgnoredIO  # type: ignore
        except ImportError as e:
            logger.warning(
                f"Failed to set Kubernetes' WSClient._all channel to _IgnoredIO: {e}"
            )
            PodOperation._failed_to_discard_duplicate_channel = True
            return
        # Whilst we can set the _all attribute whether it exists or not, we should
        # log a warning if it doesn't exist as this may indicate a change in the
        # Kubernetes library.
        if not hasattr(ws_client, "_all"):
            logger.warning(
                "Failed to set Kubernetes' WSClient._all channel to _IgnoredIO: there "
                "was no _all attribute on the WSClient object."
            )
            PodOperation._failed_to_discard_duplicate_channel = True
            return
        ws_client._all = _IgnoredIO()


def check_for_pod_restart(pod: PodInfo) -> None:
    """Check whether the pod has been replaced or its container has restarted.

    Always raises a typed exception when a change is detected; callers
    (typically ``Pod._check_for_pod_restart_sync``) are responsible for
    applying the ``restarted_container_behavior`` policy and refreshing any
    cached identity.

    Raises:
        PodReplacedError: the pod's UID has changed since ``pod.uid``.
        ContainerRestartedError: the default container's restart count has
            increased since ``pod.initial_restart_count``.
        RuntimeError: the named container is no longer present on the pod
            (treated as a permanent misconfiguration).
    """
    api = k8s_client(pod.context_name)
    snapshot = read_pod(api, name=pod.name, namespace=pod.namespace)
    if snapshot.uid != pod.uid:
        # Capture the new pod's restart count for the default container so the
        # caller can refresh its full cached identity atomically.
        raise PodReplacedError(
            pod_name=pod.name,
            old_uid=pod.uid,
            new_uid=snapshot.uid,
            new_restart_count=snapshot.restart_count_for(pod.default_container_name),
        )
    if snapshot.container_statuses is None:
        # Kubelet hasn't published container statuses yet (briefly possible
        # right after pod scheduling). Nothing to compare against — skip the
        # restart-count check.
        return
    status = snapshot.status_for(pod.default_container_name)
    if status is None:
        raise RuntimeError(
            f"Pod '{snapshot.name}' does not have a container named "
            f"'{pod.default_container_name}'"
        )
    if status.restart_count > pod.initial_restart_count:
        raise ContainerRestartedError(
            pod_name=pod.name,
            container_name=status.name,
            restart_count=status.restart_count,
            last_reason=status.last_terminated_reason or "unknown",
        )


_KEEPALIVE_PAYLOAD = json.dumps({"Width": 80, "Height": 24}).encode()  # size arbitrary


class _KeepaliveRegistration:
    """Handle for one registered websocket.

    ``unregister`` removes the socket and then waits until the shared thread
    is not mid-send, so the caller can close the socket safely.
    """

    def __init__(self, keepalive: "_SharedKeepalive", ws_client: WSClient) -> None:
        self._keepalive = keepalive
        self._ws_client = ws_client

    def unregister(self) -> None:
        self._keepalive.unregister(self._ws_client)


class _SharedKeepalive:
    """One thread that pings every live exec websocket.

    A thread per websocket does not scale: an eval with hundreds of concurrent
    sandboxes ran ~100 "ws-keepalive" threads whose only job was to sleep 30s
    at a time, and thread count growing with sandbox count starves the single
    GIL that the caller's event loop also needs (measured: 455-498 threads,
    1-2 runnable, multi-second event-loop stalls). One thread walking a
    registry sends exactly the same frames.

    ``WSClient`` is not thread-safe, so this class guarantees the invariant the
    per-socket design left implicit: a socket removed from the registry is not
    being written to once ``unregister`` returns. ``_send_lock`` is held across
    each send and by ``unregister``; the registry mutex is never held during a
    send, so registering never waits on network I/O.
    """

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._clients: list[WSClient] = []
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, ws_client: WSClient) -> _KeepaliveRegistration:
        with self._registry_lock:
            self._clients.append(ws_client)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    daemon=True,
                    name="ws-keepalive",
                )
                self._thread.start()
        return _KeepaliveRegistration(self, ws_client)

    def unregister(self, ws_client: WSClient) -> None:
        with self._registry_lock:
            try:
                self._clients.remove(ws_client)
            except ValueError:
                pass
        # Serialize with any in-flight send so the caller can close the socket.
        with self._send_lock:
            pass

    def _snapshot(self) -> list[WSClient]:
        with self._registry_lock:
            return list(self._clients)

    def _run(self) -> None:
        while True:
            self._wake.wait(_KEEPALIVE_INTERVAL_SECONDS)
            self._wake.clear()
            clients = self._snapshot()
            if not clients:
                # Exit while holding the registry lock so a concurrent
                # register() either sees a live thread or starts a new one --
                # it can never hand a socket to a thread that is exiting.
                with self._registry_lock:
                    if not self._clients:
                        self._thread = None
                        return
                continue
            for ws_client in clients:
                self._send_one(ws_client)

    def _send_one(self, ws_client: WSClient) -> None:
        """Send one keepalive frame; drop the socket if it is unusable.

        See the module comment on ``_KEEPALIVE_INTERVAL_SECONDS`` for why a
        resize-channel data frame (not a ping) is what resets the server's
        idle timer.
        """
        with self._send_lock:
            with self._registry_lock:
                if ws_client not in self._clients:
                    return  # unregistered while we waited for the send lock
            try:
                if not ws_client.is_open():
                    raise ConnectionError("websocket closed")
                ws_client.write_channel(RESIZE_CHANNEL, _KEEPALIVE_PAYLOAD)
                return
            except Exception:
                logger.debug(
                    "Failed to send k8s websocket keepalive frame, dropping socket",
                    exc_info=True,
                )
        with self._registry_lock:
            try:
                self._clients.remove(ws_client)
            except ValueError:
                pass


_KEEPALIVE = _SharedKeepalive()


def raise_for_known_read_write_errors(stderr: str) -> None:
    # The Inspect Sandbox interface asks us to raise specific exceptions for recognised
    # error messages.
    casefolded = stderr.casefold()
    if "no such file or directory" in casefolded:
        raise FileNotFoundError(stderr)
    if "permission denied" in casefolded:
        raise PermissionError(stderr)
    if "is a directory" in casefolded:
        raise IsADirectoryError(stderr)
