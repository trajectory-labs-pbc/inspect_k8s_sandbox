import threading
from time import sleep
from unittest.mock import Mock, patch

from kubernetes.stream.ws_client import RESIZE_CHANNEL  # type: ignore

from k8s_sandbox._pod.op import _SharedKeepalive


def _drain_keepalive_threads() -> None:
    """Wait out keepalive threads left over from earlier tests.

    Each test builds its own _SharedKeepalive, so a previous test's thread can
    still be finishing its idle-exit when the next one counts threads.
    """
    for _ in range(200):
        if not [t for t in threading.enumerate() if t.name == "ws-keepalive"]:
            return
        sleep(0.02)


def _open_client() -> Mock:
    ws_client = Mock()
    ws_client.is_open.return_value = True
    return ws_client


def test_sends_resize_frames_to_registered_clients():
    keepalive = _SharedKeepalive()
    ws_client = _open_client()

    with patch("k8s_sandbox._pod.op._KEEPALIVE_INTERVAL_SECONDS", 0.02):
        registration = keepalive.register(ws_client)
        sleep(0.15)
        registration.unregister()

    assert ws_client.write_channel.call_count >= 2
    for call in ws_client.write_channel.call_args_list:
        assert call[0][0] == RESIZE_CHANNEL


def test_one_thread_serves_many_clients():
    """The whole point: thread count must not scale with websocket count."""
    keepalive = _SharedKeepalive()
    _drain_keepalive_threads()
    clients = [_open_client() for _ in range(25)]
    before = {t.name for t in threading.enumerate()}

    with patch("k8s_sandbox._pod.op._KEEPALIVE_INTERVAL_SECONDS", 0.02):
        registrations = [keepalive.register(client) for client in clients]
        sleep(0.15)
        keepalive_threads = [
            t for t in threading.enumerate() if t.name == "ws-keepalive"
        ]
        for registration in registrations:
            registration.unregister()

    assert len(keepalive_threads) == 1
    for client in clients:
        assert client.write_channel.call_count >= 1
    assert {t.name for t in threading.enumerate()} - before <= {"ws-keepalive"}


def test_unregistered_client_stops_receiving_frames():
    keepalive = _SharedKeepalive()
    ws_client = _open_client()

    with patch("k8s_sandbox._pod.op._KEEPALIVE_INTERVAL_SECONDS", 0.02):
        registration = keepalive.register(ws_client)
        sleep(0.1)
        registration.unregister()
        after_unregister = ws_client.write_channel.call_count
        sleep(0.1)

    assert ws_client.write_channel.call_count == after_unregister


def test_closed_client_is_dropped_without_writes():
    keepalive = _SharedKeepalive()
    ws_client = Mock()
    ws_client.is_open.return_value = False

    with patch("k8s_sandbox._pod.op._KEEPALIVE_INTERVAL_SECONDS", 0.02):
        keepalive.register(ws_client)
        sleep(0.1)

    ws_client.write_channel.assert_not_called()
    assert keepalive._clients == []  # pyright: ignore[reportPrivateUsage]


def test_failing_client_is_dropped_but_others_keep_going():
    keepalive = _SharedKeepalive()
    broken = _open_client()
    broken.write_channel.side_effect = ConnectionResetError("boom")
    healthy = _open_client()

    with patch("k8s_sandbox._pod.op._KEEPALIVE_INTERVAL_SECONDS", 0.02):
        keepalive.register(broken)
        keepalive.register(healthy)
        sleep(0.15)
        healthy_calls = healthy.write_channel.call_count
        broken_calls = broken.write_channel.call_count

    assert broken_calls == 1, "a failing socket must not be retried"
    assert healthy_calls >= 2, "one bad socket must not stall the shared thread"


def test_unregister_waits_for_in_flight_send():
    """unregister() must not return while a send is in progress.

    WSClient is not thread-safe, so the caller closing the socket immediately
    after unregister() would otherwise race the keepalive thread mid-write.
    """
    keepalive = _SharedKeepalive()
    in_send = threading.Event()
    release = threading.Event()
    ws_client = _open_client()

    def blocking_write(*_args: object) -> None:
        in_send.set()
        release.wait(timeout=5)

    ws_client.write_channel.side_effect = blocking_write

    with patch("k8s_sandbox._pod.op._KEEPALIVE_INTERVAL_SECONDS", 0.02):
        registration = keepalive.register(ws_client)
        assert in_send.wait(timeout=5), "keepalive never entered the send"

        unregistered = threading.Event()

        def unregister() -> None:
            registration.unregister()
            unregistered.set()

        waiter = threading.Thread(target=unregister)
        waiter.start()
        assert not unregistered.wait(timeout=0.2), (
            "unregister returned while a send was in flight"
        )
        release.set()
        assert unregistered.wait(timeout=5)
        waiter.join(timeout=5)


def test_thread_exits_when_registry_empties():
    """An idle keepalive thread must not outlive its last websocket."""
    keepalive = _SharedKeepalive()
    _drain_keepalive_threads()
    ws_client = _open_client()

    with patch("k8s_sandbox._pod.op._KEEPALIVE_INTERVAL_SECONDS", 0.02):
        registration = keepalive.register(ws_client)
        sleep(0.05)
        registration.unregister()
        for _ in range(200):
            if keepalive._thread is None:  # pyright: ignore[reportPrivateUsage]
                break
            sleep(0.02)

    assert keepalive._thread is None  # pyright: ignore[reportPrivateUsage]

    # ...and a later register() must start a fresh one.
    with patch("k8s_sandbox._pod.op._KEEPALIVE_INTERVAL_SECONDS", 0.02):
        second = _open_client()
        registration = keepalive.register(second)
        sleep(0.1)
        registration.unregister()
    assert second.write_channel.call_count >= 1
