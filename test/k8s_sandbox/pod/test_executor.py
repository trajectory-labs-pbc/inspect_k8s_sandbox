import asyncio
import contextvars
import json
import logging
import math
import threading
from time import sleep
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from pytest import MonkeyPatch

import k8s_sandbox._pod.op as op_module
from k8s_sandbox._pod.executor import PodOpExecutor
from k8s_sandbox._pod.op import PodInfo, PodOperation


@pytest.fixture(autouse=True)
def reset_singleton() -> Generator:
    # Ensure that each test starts with a fresh singleton instance.
    PodOpExecutor._instance = None
    yield


def test_get_instance() -> None:
    result1 = PodOpExecutor.get_instance()
    result2 = PodOpExecutor.get_instance()

    assert result1 == result2


def test_default_max_workers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("INSPECT_MAX_POD_OPS", raising=False)

    with patch("os.cpu_count", return_value=4):
        actual = PodOpExecutor.get_instance()

    assert actual._max_workers == 16


def test_max_workers_via_env_var(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_MAX_POD_OPS", "42")

    actual = PodOpExecutor.get_instance()

    assert actual._max_workers == 42


def test_max_workers_via_parameter(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("INSPECT_MAX_POD_OPS", raising=False)

    actual = PodOpExecutor.get_instance(max_pod_ops=64)

    assert actual._max_workers == 64


def test_parameter_takes_precedence_over_env_var(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_MAX_POD_OPS", "42")

    actual = PodOpExecutor.get_instance(max_pod_ops=64)

    assert actual._max_workers == 64


def test_parameter_conflicting_with_existing_executor_raises(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("INSPECT_MAX_POD_OPS", raising=False)

    with patch("os.cpu_count", return_value=4):
        PodOpExecutor.get_instance()

    with pytest.raises(ValueError, match="already initialized with max_pod_ops=16"):
        PodOpExecutor.get_instance(max_pod_ops=64)


async def test_queue_operation(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_MAX_POD_OPS", "10")
    executor = PodOpExecutor.get_instance()

    op1 = executor.queue_operation(lambda: _synchronous_operation(1))
    op2 = executor.queue_operation(lambda: _synchronous_operation(2))

    result1, result2 = await asyncio.gather(op1, op2)
    assert result1 == (1, "pod-op-executor_0")
    assert result2 == (2, "pod-op-executor_1")


async def test_queue_more_operations_than_max_workers(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_MAX_POD_OPS", "2")
    executor = PodOpExecutor.get_instance()

    op1 = executor.queue_operation(lambda: _synchronous_operation(1))
    op2 = executor.queue_operation(lambda: _synchronous_operation(2))
    op3 = executor.queue_operation(lambda: _synchronous_operation(3))

    result1, result2, result3 = await asyncio.gather(op1, op2, op3)
    assert result1 == (1, "pod-op-executor_0")
    assert result2 == (2, "pod-op-executor_1")
    # The third operation should be executed by one of the two existing workers.
    assert result3 == (3, "pod-op-executor_0") or result3 == (3, "pod-op-executor_1")


async def test_queue_operation_propagates_caller_context(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("INSPECT_MAX_POD_OPS", "1")
    executor = PodOpExecutor.get_instance()
    var: contextvars.ContextVar[str] = contextvars.ContextVar(
        "test_var", default="default"
    )
    var.set("override")

    seen = await executor.queue_operation(var.get)

    assert seen == "override"


async def test_slow_operation_logs_stream_setup_and_command_durations(
    monkeypatch: MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("INSPECT_POD_OP_SLOW_SECONDS", "0")
    executor = PodOpExecutor.get_instance(max_pod_ops=1)
    websocket = MagicMock()
    registration = MagicMock()
    operation = PodOperation(
        PodInfo(
            name="pod",
            namespace="namespace",
            context_name=None,
            default_container_name="container",
            uid="uid",
            initial_restart_count=0,
            restarted_container_behavior="raise",
        )
    )
    monkeypatch.setattr(op_module, "k8s_client", MagicMock())
    stream_factory = MagicMock(return_value=websocket)
    monkeypatch.setattr(op_module, "stream", stream_factory)
    monkeypatch.setattr(
        op_module._KEEPALIVE, "register", MagicMock(return_value=registration)
    )
    monkeypatch.setattr(operation, "_discard_duplicate_channel", MagicMock())

    def execute_stream() -> None:
        websocket_stream = operation.create_websocket_client_for_exec(command=["true"])
        next(websocket_stream)
        sleep(0.01)
        websocket_stream.close()

    caplog.set_level(logging.WARNING, logger="k8s_sandbox._logger")

    await executor.queue_operation(execute_stream)

    slow_operation = next(
        record for record in caplog.records if "Slow pod operation." in record.message
    )
    _, serialized_fields = slow_operation.message.split(
        "Slow pod operation. ", maxsplit=1
    )
    fields = json.loads(serialized_fields)

    assert stream_factory.called
    assert registration.unregister.called
    assert websocket.close.called
    connect_seconds = float(fields["connect_s"])
    command_seconds = float(fields["command_s"])
    call_seconds = float(fields["call_s"])
    assert connect_seconds >= 0
    assert command_seconds >= 0
    assert math.isclose(
        connect_seconds + command_seconds,
        call_seconds,
        abs_tol=0.01,
    )


def _synchronous_operation(value: int) -> tuple[int, str]:
    sleep(1)
    return value, threading.current_thread().name
