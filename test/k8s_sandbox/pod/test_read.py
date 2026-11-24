import io
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator
from unittest.mock import MagicMock

import pytest
from kubernetes.stream.ws_client import (  # type: ignore[import-untyped]
    ERROR_CHANNEL,
    WSClient,
)
from websocket import WebSocketConnectionClosedException

import k8s_sandbox._pod.op as op_module
import k8s_sandbox._pod.read as readfile_module
from k8s_sandbox._pod.error import ContainerStoppedError, PodError
from k8s_sandbox._pod.read import ReadFileOperation
from test.k8s_sandbox.pod.test_execute import _FakeWSClient, _status


def _exec_kwargs(monkeypatch: pytest.MonkeyPatch, src: Path) -> dict[str, Any]:
    """The arguments read_file() opens its exec with."""
    op = ReadFileOperation(MagicMock())
    exec_kwargs: dict[str, Any] = {}

    def fake_exec(**kwargs: Any) -> Generator[MagicMock, None, None]:
        exec_kwargs.update(kwargs)
        yield MagicMock(spec=WSClient)

    monkeypatch.setattr(op, "create_websocket_client_for_exec", fake_exec)
    monkeypatch.setattr(op, "_handle_stream_output", lambda ws_client, dst: None)

    op.read_file(src, io.BytesIO())
    return exec_kwargs


def test_read_file_opens_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    # A read exec without stdin hangs or truncates on Kata Containers' Go shim
    # (kata-containers/kata-containers#9071).
    exec_kwargs = _exec_kwargs(monkeypatch, Path("/tmp/src"))

    assert exec_kwargs["stdin"] is True
    assert exec_kwargs["command"][-1] == "/tmp/src"


def test_read_file_command_is_all_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    # kubernetes>=37 validates each element and rejects the call before sending it.
    op = ReadFileOperation(MagicMock())
    exec_kwargs: dict[str, Any] = {}

    def fake_exec(**kwargs: Any) -> Generator[MagicMock, None, None]:
        exec_kwargs.update(kwargs)
        yield MagicMock(spec=WSClient)

    monkeypatch.setattr(op, "create_websocket_client_for_exec", fake_exec)
    monkeypatch.setattr(op, "_handle_stream_output", lambda ws_client, dst: None)

    op.read_file(Path("/tmp/src"), io.BytesIO())

    command = exec_kwargs["command"]
    assert all(isinstance(arg, str) for arg in command), command


@pytest.mark.parametrize(
    ("returncode", "error"),
    [
        # `head` and `mkdir` never exit 255: Kata's status for a stopped VM.
        pytest.param(255, ContainerStoppedError, id="container-stopped"),
        pytest.param(2, PodError, id="failed"),
    ],
)
def test_a_255_is_a_stopped_container(
    monkeypatch: pytest.MonkeyPatch, returncode: int, error: type[Exception]
) -> None:
    monkeypatch.setattr(readfile_module, "get_returncode", lambda _ws: returncode)
    op = ReadFileOperation(MagicMock())
    ws = MagicMock(spec=WSClient)
    ws.is_open.return_value = False
    ws.read_stderr.return_value = ""

    with pytest.raises(error) as raised:
        op._handle_stream_output(ws, io.BytesIO())

    assert type(raised.value) is error


@pytest.mark.parametrize(
    "lost",
    [ConnectionResetError(), WebSocketConnectionClosedException()],
    ids=["reset", "closed"],
)
@pytest.mark.parametrize(
    ("frames", "stopped"),
    [
        ([_status(255)], True),
        ([_status(2)], False),
        # As for an exec into a dead container: no exit code, nor even `details`.
        ([(ERROR_CHANNEL, b'{"status":"Failure","reason":"InternalError"}')], False),
        ([], False),
    ],
    ids=["255", "another-status", "no-exit-code", "no-status"],
)
def test_a_255_before_the_connection_is_lost_is_a_stopped_container(
    frames: list[tuple[int, bytes]], stopped: bool, lost: Exception
) -> None:
    ws = _FakeWSClient(frames, error=lost)

    with pytest.raises(Exception) as raised:
        ReadFileOperation(MagicMock())._handle_stream_output(ws, io.BytesIO())

    assert type(raised.value) is (ContainerStoppedError if stopped else type(lost))


@pytest.mark.skipif(os.name != "posix", reason="Requires a POSIX shell")
def test_a_path_that_looks_like_an_option_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "--help").write_bytes(b"contents")
    command = _exec_kwargs(monkeypatch, Path("--help"))["command"]

    result = subprocess.run(command, cwd=tmp_path, capture_output=True, timeout=30)

    assert result.stdout == b"contents"


@pytest.mark.skipif(os.name != "posix", reason="Requires a POSIX shell")
def test_head_is_not_found_through_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """It runs as the container's user."""
    ran = tmp_path / "ran"
    stub = tmp_path / "head"
    stub.write_text(f"#!/bin/sh\necho head >> {ran}\n")
    stub.chmod(0o755)
    src = tmp_path / "src"
    src.write_bytes(b"contents")
    command = _exec_kwargs(monkeypatch, src)["command"]
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}

    result = subprocess.run(command, env=env, capture_output=True, timeout=30)

    assert not ran.exists()
    assert result.stdout == b"contents"


def _finish_within(seconds: float, fn: Callable[[], object]) -> object:
    """Run fn on a worker thread; fail if it is still waiting after `seconds`."""
    outcome: list[object] = []

    def run() -> None:
        try:
            outcome.append(fn())
        except BaseException as e:  # noqa: BLE001 - the test inspects the exception
            outcome.append(e)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        pytest.fail(f"still waiting on a pod that will never answer after {seconds}s")
    return outcome[0]


class TestPodStopsAnswering:
    """`head` on a live pod streams continuously; a long silence means it is gone."""

    def test_read_file_raises_when_the_stream_stalls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(readfile_module, "READ_STALL_SECONDS", 0.2)
        monkeypatch.setattr(op_module, "TRANSPORT_POLL_SECONDS", 0.05)
        ws = MagicMock(spec=WSClient)
        ws.is_open.return_value = True

        def update(timeout: float | None = None) -> None:
            time.sleep(min(timeout if timeout is not None else 60.0, 60.0))

        ws.update.side_effect = update
        ws.peek_stdout.return_value = False
        ws.peek_stderr.return_value = False
        reader = ReadFileOperation(MagicMock())

        outcome = _finish_within(
            5.0, lambda: reader._handle_stream_output(ws, io.BytesIO())
        )

        assert isinstance(outcome, PodError)
        assert "stalled" in str(outcome)

    def test_read_file_tolerates_a_slow_stream_that_keeps_delivering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bytes keep arriving, each gap shorter than the stall bound: no error."""
        monkeypatch.setattr(readfile_module, "READ_STALL_SECONDS", 0.4)
        monkeypatch.setattr(op_module, "TRANSPORT_POLL_SECONDS", 0.05)
        monkeypatch.setattr(readfile_module, "get_returncode", lambda ws: 0)
        ws = MagicMock(spec=WSClient)
        frames = [b"a", b"b", b"c", b"d", b"e"]
        delivered = 0
        ws.is_open.side_effect = lambda: delivered < len(frames)

        def update(timeout: float | None = None) -> None:
            # Slower than the poll interval, faster than the stall bound; the total
            # transfer (0.75s) is longer than the stall bound, so only per-byte
            # progress can keep this read alive.
            time.sleep(0.15)

        def read_stdout() -> bytes:
            nonlocal delivered
            frame = frames[delivered]
            delivered += 1
            return frame

        ws.update.side_effect = update
        ws.peek_stdout.side_effect = lambda: delivered < len(frames)
        ws.read_stdout.side_effect = read_stdout
        ws.peek_stderr.return_value = False
        dst = io.BytesIO()

        ReadFileOperation(MagicMock())._handle_stream_output(ws, dst)

        assert dst.getvalue() == b"abcde"
