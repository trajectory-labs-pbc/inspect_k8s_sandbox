"""Exercise exec output handling when a websocket-like client returns None.

The stub presents a truthy peek followed by None from read. This documents the
behavioral precondition exercised by these tests.
"""

import json
from typing import Any

import pytest

from k8s_sandbox._pod.execute import COMPLETED_SENTINEL, ExecuteOperation


class _RaceyWSClient:
    """Stub whose scripted reads can return None after truthy peeks.

    Frame script per channel: each entry is either bytes or None.
    """

    def __init__(
        self,
        stdout_frames: list[bytes | None],
        stderr_frames: list[bytes | None] | None = None,
    ) -> None:
        self._stdout = list(stdout_frames)
        self._stderr = list(stderr_frames or [])
        self._open = True
        self.returncode = 0

    def is_open(self) -> bool:
        return self._open and bool(self._stdout or self._stderr)

    def update(self, timeout: Any = None) -> None:  # noqa: ANN401
        return None

    def peek_stdout(self, timeout: Any = None) -> bool:  # noqa: ANN401
        return bool(self._stdout)

    def read_stdout(self, timeout: Any = None) -> bytes | None:  # noqa: ANN401
        return self._stdout.pop(0) if self._stdout else None

    def peek_stderr(self, timeout: Any = None) -> bool:  # noqa: ANN401
        return bool(self._stderr)

    def read_stderr(self, timeout: Any = None) -> bytes | None:  # noqa: ANN401
        return self._stderr.pop(0) if self._stderr else None

    def close(self) -> None:
        self._open = False

    def read_channel(self, channel: int, timeout: Any = None) -> str:  # noqa: ANN401
        """k8s ERROR_CHANNEL (status channel) used to derive the exit code."""
        if self.returncode == 0:
            return json.dumps({"metadata": {}, "status": "Success"})
        return json.dumps(
            {
                "metadata": {},
                "status": "Failure",
                "message": "command terminated with non-zero exit code",
                "details": {
                    "causes": [{"reason": "ExitCode", "message": str(self.returncode)}]
                },
            }
        )


def _run(client: _RaceyWSClient) -> Any:  # noqa: ANN401
    op = object.__new__(ExecuteOperation)
    return ExecuteOperation._handle_shell_output(op, client, None, None)  # type: ignore[arg-type]


def _sentinel_frame(returncode: int = 0) -> bytes:
    """The real in-band completion marker the loop parses out of stdout."""
    return f"<{COMPLETED_SENTINEL}-{returncode}>".encode()


def test_none_stdout_frame_does_not_raise_attributeerror() -> None:
    """A None stdout read does not raise AttributeError."""
    client = _RaceyWSClient(
        stdout_frames=[b"hello ", None, b"world", _sentinel_frame(0)]
    )

    try:
        result = _run(client)
    except AttributeError as e:  # pragma: no cover - this is the bug
        pytest.fail(f"None stdout frame reached the decoder: {e}.")

    assert "hello" in result.stdout
    assert "world" in result.stdout, "output either side of the None must survive"


def test_none_stderr_frame_does_not_poison_the_buffer() -> None:
    # stdout yields empty filler so the loop keeps iterating while stderr
    # drains; the sentinel must come LAST because it closes the client.
    client = _RaceyWSClient(
        stdout_frames=[b"", b"", b"", _sentinel_frame(0)],
        stderr_frames=[b"warn ", None, b"more"],
    )

    result = _run(client)

    assert "warn" in result.stderr
    assert "more" in result.stderr, "output after the None frame must survive"


def test_all_none_stdout_frames_still_terminate() -> None:
    """A channel that only yields None terminates without spinning or crashing."""
    client = _RaceyWSClient(stdout_frames=[None, None, _sentinel_frame(3)])
    client.returncode = 3

    result = _run(client)

    assert result.returncode == 3
